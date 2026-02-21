import asyncio
import os
import gc
import json
import torch
import torch._dynamo
import torchaudio
import logging
from typing import Optional, Callable, Union, Dict
from tqdm import tqdm
from backend.app.models import GenerationRequest, Job, JobStatus
from sqlmodel import Session, select
from tokenizers import Tokenizer

# Try mmgp pipeline first (more reliable on 12GB GPUs)
try:
    from backend.heartmula import HeartMuLaPipeline
    from mmgp import offload as mmgp_offload
    MMGP_AVAILABLE = True
except ImportError:
    MMGP_AVAILABLE = False

# Fallback to heartlib pipeline
try:
    from heartlib.pipelines.music_generation import HeartMuLaGenPipeline, HeartMuLaGenConfig
    from heartlib.heartmula.modeling_heartmula import HeartMuLa
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec
    HEARTLIB_AVAILABLE = True
except ImportError:
    HEARTLIB_AVAILABLE = False
    HeartMuLaGenPipeline = None
    HeartMuLaGenConfig = None
    HeartMuLa = None
    HeartCodec = None

# Optional: 4-bit quantization support (for heartlib mode)
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

# HuggingFace Hub for auto-downloading models
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

logger = logging.getLogger(__name__)

# HuggingFace model IDs (base repos - version is appended for HeartMuLa)
HF_HEARTCODEC_REPO = "HeartMuLa/HeartCodec-oss-20260123"
HF_HEARTMULA_GEN_REPO = "HeartMuLa/HeartMuLaGen"  # Contains tokenizer.json and gen_config.json

# Model version mapping - maps version to (HuggingFace repo, local folder name)
# Local folder name must match heartlib's expected format: HeartMuLa-oss-{version}
MODEL_VERSIONS = {
    "3B": ("HeartMuLa/HeartMuLa-oss-3B", "HeartMuLa-oss-3B"),
    "RL-3B-20260123": ("HeartMuLa/HeartMuLa-RL-oss-3B-20260123", "HeartMuLa-oss-RL-3B-20260123"),
    "3B-happy-new-year": ("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", "HeartMuLa-oss-3B-happy-new-year"),
}

# Default version to use - best open-source model for lyrics controllability and music quality
DEFAULT_VERSION = os.environ.get("HEARTMULA_VERSION", "3B-happy-new-year")

# Configuration: GPU mode settings
# These can be set manually via environment variables, or left as "auto" for automatic detection
# HEARTMULA_USE_MMGP: "true", "false", or "auto" (default: auto) - Use mmgp memory management (recommended for 12GB GPUs)
# HEARTMULA_4BIT: "true", "false", or "auto" (default: auto) - Only used if mmgp is disabled
# HEARTMULA_SEQUENTIAL_OFFLOAD: "true", "false", or "auto" (default: auto)
# HEARTMULA_COMPILE: "true" or "false" (default: false) - Enable torch.compile for ~2x faster inference
# HEARTMULA_COMPILE_MODE: "default", "reduce-overhead", or "max-autotune" (default: default)
_MMGP_ENV = os.environ.get("HEARTMULA_USE_MMGP", "auto").lower()
_4BIT_ENV = os.environ.get("HEARTMULA_4BIT", "auto").lower()
_OFFLOAD_ENV = os.environ.get("HEARTMULA_SEQUENTIAL_OFFLOAD", "auto").lower()
_COMPILE_ENV = os.environ.get("HEARTMULA_COMPILE", "false").lower()
_COMPILE_MODE_ENV = os.environ.get("HEARTMULA_COMPILE_MODE", "default").lower()

# Determine if mmgp should be used
USE_MMGP = None  # None = auto-detect
if _MMGP_ENV == "true":
    USE_MMGP = True
elif _MMGP_ENV == "false":
    USE_MMGP = False
# else: auto - will be determined based on VRAM and availability

# Manual overrides (if explicitly set to true/false)
ENABLE_4BIT_QUANTIZATION = _4BIT_ENV == "true" if _4BIT_ENV != "auto" else None
ENABLE_SEQUENTIAL_OFFLOAD = _OFFLOAD_ENV == "true" if _OFFLOAD_ENV != "auto" else None
ENABLE_TORCH_COMPILE = _COMPILE_ENV == "true"
TORCH_COMPILE_MODE = _COMPILE_MODE_ENV if _COMPILE_MODE_ENV in ["default", "reduce-overhead", "max-autotune"] else "default"

# VRAM thresholds for auto-detection (in GB)
VRAM_THRESHOLD_FULL_PRECISION = 20.0  # Can fit HeartMuLa (~11GB) + HeartCodec (~6GB) + KV cache (~4GB)
VRAM_THRESHOLD_QUANTIZED_NO_SWAP = 14.0  # Can fit 4-bit HeartMuLa (~3GB) + HeartCodec (~6GB) + KV cache (~4GB)
VRAM_THRESHOLD_QUANTIZED_WITH_SWAP = 10.0  # Can fit 4-bit HeartMuLa (~3GB) + KV cache (~4GB), then swap for codec
VRAM_MINIMUM = 8.0  # Absolute minimum to run at all

# Model directory - configurable via HEARTMULA_MODEL_DIR env var
# Allows users to point to existing models (e.g., ComfyUI shared models)
_default_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
DEFAULT_MODEL_DIR = os.environ.get("HEARTMULA_MODEL_DIR", _default_model_dir)

# Settings persistence file - in db directory which is mounted as a volume
_default_db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "db")
SETTINGS_FILE = os.path.join(os.environ.get("HEARTMULA_DB_PATH", _default_db_dir).replace("jobs.db", ""), "settings.json")


def detect_optimal_gpu_config() -> dict:
    """
    Auto-detect the optimal GPU configuration based on available VRAM.

    Returns a dict with:
        - use_quantization: bool - whether to use 4-bit quantization (legacy, for heartlib mode)
        - use_sequential_offload: bool - whether to use lazy codec loading with model swapping
        - use_mmgp: bool - whether mmgp memory management should be used
        - num_gpus: int - number of GPUs detected
        - gpu_info: dict - info about each GPU (name, vram, compute capability)
        - config_name: str - human-readable name of the selected configuration
        - warning: str or None - any warnings about the configuration
    """
    result = {
        "use_quantization": True,  # Legacy: for BitsAndBytes mode
        "use_sequential_offload": True,  # Default to sequential for safety
        "use_mmgp": MMGP_AVAILABLE,  # Prefer mmgp when available
        "num_gpus": 0,
        "gpu_info": {},
        "config_name": "CPU Only",
        "warning": None,
    }

    if not torch.cuda.is_available():
        result["warning"] = "No CUDA GPU detected. Running on CPU will be very slow."
        return result

    num_gpus = torch.cuda.device_count()
    result["num_gpus"] = num_gpus

    # Gather GPU info
    gpu_info = {}
    total_vram = 0
    max_vram = 0
    max_vram_gpu = 0
    max_compute = 0
    max_compute_gpu = 0

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_vram_gb = props.total_memory / (1024 ** 3)
        # Get available/free VRAM (total - already allocated)
        with torch.cuda.device(i):
            free_vram_bytes = torch.cuda.mem_get_info()[0]  # (free, total)
            free_vram_gb = free_vram_bytes / (1024 ** 3)
        compute_cap = props.major + props.minor / 10
        gpu_info[i] = {
            "name": props.name,
            "vram_gb": total_vram_gb,
            "vram_free_gb": free_vram_gb,
            "compute_capability": compute_cap,
            "supports_flash_attention": compute_cap >= 7.0,
        }
        total_vram += total_vram_gb
        # Use FREE VRAM for decision making, not total
        if free_vram_gb > max_vram:
            max_vram = free_vram_gb
            max_vram_gpu = i
        if compute_cap > max_compute:
            max_compute = compute_cap
            max_compute_gpu = i

    result["gpu_info"] = gpu_info

    # Log detected GPUs with both total and available VRAM
    print(f"\n[Auto-Config] Detected {num_gpus} GPU(s):", flush=True)
    for i, info in gpu_info.items():
        fa_status = "✓ Flash Attention" if info["supports_flash_attention"] else "✗ No Flash Attention"
        vram_status = f"{info['vram_free_gb']:.1f}GB free / {info['vram_gb']:.1f}GB total"
        print(f"  GPU {i}: {info['name']} ({vram_status}, SM {info['compute_capability']}) - {fa_status}", flush=True)

    # Decision logic for single GPU
    if num_gpus == 1:
        # Use FREE VRAM for threshold decisions (accounts for VRAM used by other apps)
        vram = gpu_info[0]["vram_free_gb"]
        total_vram = gpu_info[0]["vram_gb"]
        print(f"[Auto-Config] Using FREE VRAM ({vram:.1f}GB) for configuration (total: {total_vram:.1f}GB)", flush=True)

        if vram >= VRAM_THRESHOLD_FULL_PRECISION:
            # 20GB+: Full precision, no swapping needed
            result["use_quantization"] = False
            result["use_sequential_offload"] = False
            result["config_name"] = "Full Precision (Single GPU)"
            print(f"[Auto-Config] Selected: FULL PRECISION mode ({vram:.1f}GB VRAM >= {VRAM_THRESHOLD_FULL_PRECISION}GB threshold)", flush=True)

        elif vram >= VRAM_THRESHOLD_QUANTIZED_NO_SWAP:
            # 14-20GB: Can fit both models with mmgp bf16
            result["use_quantization"] = False
            result["use_sequential_offload"] = False
            if MMGP_AVAILABLE:
                result["config_name"] = "mmgp bf16 (Single GPU)"
                print(f"[Auto-Config] Selected: MMGP BF16 mode ({vram:.1f}GB VRAM)", flush=True)
            else:
                result["config_name"] = "4-bit Quantized (Single GPU)"
                result["use_quantization"] = True
                print(f"[Auto-Config] Selected: 4-BIT QUANTIZED mode ({vram:.1f}GB VRAM)", flush=True)

        elif vram >= VRAM_THRESHOLD_QUANTIZED_WITH_SWAP:
            # 10-14GB: Use mmgp with model swapping (bf16, no BitsAndBytes)
            result["use_quantization"] = True  # Legacy flag
            result["use_sequential_offload"] = True
            if MMGP_AVAILABLE:
                result["config_name"] = "mmgp bf16 + Model Swap (Single GPU)"
                print(f"[Auto-Config] Selected: MMGP BF16 + MODEL SWAP mode ({vram:.1f}GB VRAM)", flush=True)
                print(f"[Auto-Config] Note: Models swap between transformer/codec phases.", flush=True)
            else:
                result["config_name"] = "4-bit Quantized + Sequential Offload (Single GPU)"
                print(f"[Auto-Config] Selected: 4-BIT QUANTIZED + SEQUENTIAL OFFLOAD mode ({vram:.1f}GB VRAM)", flush=True)
                print(f"[Auto-Config] Note: Models will be swapped in/out of VRAM.", flush=True)

        elif vram >= VRAM_MINIMUM:
            # 8-10GB: Aggressive mode
            result["use_quantization"] = True
            result["use_sequential_offload"] = True
            if MMGP_AVAILABLE:
                result["config_name"] = "mmgp bf16 + Model Swap (Low VRAM)"
                result["warning"] = f"Low VRAM ({vram:.1f}GB). May encounter OOM errors."
                print(f"[Auto-Config] WARNING: Low VRAM ({vram:.1f}GB). Using mmgp with aggressive swapping.", flush=True)
            else:
                result["config_name"] = "4-bit Quantized + Sequential Offload (Low VRAM)"
                result["warning"] = f"Low VRAM ({vram:.1f}GB). Generation may fail or be very slow."
                print(f"[Auto-Config] WARNING: Low VRAM ({vram:.1f}GB). Using 4-bit + sequential offload.", flush=True)

        else:
            # <8GB: Probably won't work
            result["use_quantization"] = True
            result["use_sequential_offload"] = True
            result["config_name"] = "Insufficient VRAM"
            result["warning"] = f"Insufficient VRAM ({vram:.1f}GB < {VRAM_MINIMUM}GB minimum). Generation will likely fail."
            print(f"[Auto-Config] ERROR: Insufficient VRAM ({vram:.1f}GB). Minimum {VRAM_MINIMUM}GB required.", flush=True)

    # Decision logic for multi-GPU
    else:
        # Multi-GPU: Distribute models across GPUs
        # HeartMuLa goes to fastest GPU (for Flash Attention)
        # HeartCodec goes to GPU with most VRAM

        mula_gpu = max_compute_gpu
        codec_gpu = max_vram_gpu

        # If same GPU is both fastest and largest, put codec on second-best
        if mula_gpu == codec_gpu:
            # Find second GPU with most VRAM
            second_vram = 0
            for i, info in gpu_info.items():
                if i != mula_gpu and info["vram_gb"] > second_vram:
                    second_vram = info["vram_gb"]
                    codec_gpu = i

        mula_vram = gpu_info[mula_gpu]["vram_gb"]
        codec_vram = gpu_info[codec_gpu]["vram_gb"]

        # Check if we need quantization for the HeartMuLa GPU
        if mula_vram >= 16.0:
            # HeartMuLa GPU has enough VRAM for full precision
            result["use_quantization"] = False
            result["config_name"] = f"Full Precision Multi-GPU (HeartMuLa: GPU {mula_gpu}, HeartCodec: GPU {codec_gpu})"
            print(f"[Auto-Config] Selected: FULL PRECISION MULTI-GPU mode", flush=True)
        else:
            # Need quantization for HeartMuLa
            result["use_quantization"] = True
            result["config_name"] = f"4-bit Multi-GPU (HeartMuLa: GPU {mula_gpu}, HeartCodec: GPU {codec_gpu})"
            print(f"[Auto-Config] Selected: 4-BIT MULTI-GPU mode", flush=True)

        # Multi-GPU never needs sequential offload (models on different GPUs)
        result["use_sequential_offload"] = False

        print(f"[Auto-Config] HeartMuLa -> GPU {mula_gpu}: {gpu_info[mula_gpu]['name']} ({mula_vram:.1f}GB)", flush=True)
        print(f"[Auto-Config] HeartCodec -> GPU {codec_gpu}: {gpu_info[codec_gpu]['name']} ({codec_vram:.1f}GB)", flush=True)

        # Check if codec GPU has enough VRAM
        if codec_vram < 8.0:
            result["warning"] = f"HeartCodec GPU has low VRAM ({codec_vram:.1f}GB). May encounter issues."

    print(f"[Auto-Config] Configuration: {result['config_name']}", flush=True)
    if result["warning"]:
        print(f"[Auto-Config] ⚠️  {result['warning']}", flush=True)
    print("", flush=True)

    return result


def ensure_models_downloaded(model_dir: str = DEFAULT_MODEL_DIR, version: str = None) -> str:
    """
    Ensure HeartMuLa models are downloaded. Downloads from HuggingFace Hub if not present.

    Args:
        model_dir: Directory to store/find models
        version: Model version (e.g., "3B", "RL-3B-20260123"). If None, uses DEFAULT_VERSION.

    Returns the path to the model directory.
    """
    if version is None:
        version = DEFAULT_VERSION

    os.makedirs(model_dir, exist_ok=True)

    # Get HuggingFace repo ID and local folder name for this version
    if version in MODEL_VERSIONS:
        hf_repo, folder_name = MODEL_VERSIONS[version]
    else:
        # Fallback: assume version is the folder suffix
        hf_repo = f"HeartMuLa/HeartMuLa-oss-{version}"
        folder_name = f"HeartMuLa-oss-{version}"

    heartmula_path = os.path.join(model_dir, folder_name)
    heartcodec_path = os.path.join(model_dir, "HeartCodec-oss")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    gen_config_path = os.path.join(model_dir, "gen_config.json")

    all_present = (
        os.path.exists(heartmula_path) and
        os.path.exists(heartcodec_path) and
        os.path.isfile(tokenizer_path) and
        os.path.isfile(gen_config_path)
    )

    if all_present:
        logger.info(f"All models found at {model_dir}")
        print(f"[Models] All models found at {model_dir}", flush=True)
        return model_dir

    if not HF_HUB_AVAILABLE:
        raise RuntimeError(
            "Models not found and huggingface_hub is not installed. "
            "Please install it with: pip install huggingface_hub\n"
            "Or manually download models to: " + model_dir
        )

    print(f"[Models] Downloading models from HuggingFace Hub to {model_dir}...", flush=True)
    print(f"[Models] This may take a while on first run (~5GB download)...", flush=True)

    # Download HeartMuLa model
    if not os.path.exists(heartmula_path):
        print(f"[Models] Downloading {hf_repo}...", flush=True)
        snapshot_download(
            repo_id=hf_repo,
            local_dir=heartmula_path,
            local_dir_use_symlinks=False,
        )
        print(f"[Models] {folder_name} downloaded.", flush=True)

    # Download HeartCodec model
    if not os.path.exists(heartcodec_path):
        print(f"[Models] Downloading HeartCodec-oss...", flush=True)
        snapshot_download(
            repo_id=HF_HEARTCODEC_REPO,
            local_dir=heartcodec_path,
            local_dir_use_symlinks=False,
        )
        print(f"[Models] HeartCodec-oss downloaded.", flush=True)

    # Download tokenizer and gen_config from HeartMuLaGen repo
    if not os.path.isfile(tokenizer_path):
        print(f"[Models] Downloading tokenizer.json...", flush=True)
        hf_hub_download(
            repo_id=HF_HEARTMULA_GEN_REPO,
            filename="tokenizer.json",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )

    if not os.path.isfile(gen_config_path):
        print(f"[Models] Downloading gen_config.json...", flush=True)
        hf_hub_download(
            repo_id=HF_HEARTMULA_GEN_REPO,
            filename="gen_config.json",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )

    print(f"[Models] All models downloaded successfully!", flush=True)
    logger.info(f"Models downloaded to {model_dir}")
    return model_dir


def apply_torch_compile(model, compile_mode: str = "default"):
    """
    Apply torch.compile to HeartMuLa model for faster inference.

    This can provide ~2x speedup on supported GPUs (tested on RTX 4090, A100).
    First run will be slower due to compilation, but subsequent runs are faster.

    Requirements:
    - Turing (SM 7.5) or newer GPU recommended
    - Triton with inductor backend for best performance

    Args:
        model: HeartMuLa model instance
        compile_mode: One of "default", "reduce-overhead", or "max-autotune"

    Returns:
        The compiled model, or original model if compilation fails
    """
    if not ENABLE_TORCH_COMPILE:
        return model

    # Check GPU compute capability - torch.compile with Triton works best on SM 7.5+
    # Older GPUs (Pascal SM 6.x, Volta SM 7.0) may have issues or be very slow
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        compute_cap = props.major + props.minor / 10
        if compute_cap < 7.5:
            print(f"[torch.compile] WARNING: GPU {props.name} (SM {compute_cap}) is older than recommended.", flush=True)
            print(f"[torch.compile] torch.compile works best on Turing (SM 7.5) or newer GPUs.", flush=True)
            print(f"[torch.compile] On older GPUs, compilation can be very slow or may fail.", flush=True)
            print(f"[torch.compile] Auto-disabling torch.compile for better stability.", flush=True)
            return model

    try:
        # Check if triton is available for optimal performance
        try:
            import triton
            # Test if triton can actually compile (needs C compiler)
            import subprocess
            result = subprocess.run(['which', 'gcc'], capture_output=True)
            if result.returncode != 0:
                result = subprocess.run(['which', 'cc'], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError("No C compiler found (gcc/cc). Triton inductor backend requires a C compiler.")
            backend = "inductor"
            print(f"[torch.compile] Triton found with C compiler - using inductor backend", flush=True)
        except ImportError:
            import warnings
            warnings.warn(
                "Triton not found. On Windows, install triton-windows for best performance: "
                "pip install -U 'triton-windows>=3.2,<3.3'. Falling back to eager backend."
            )
            backend = "eager"
            print(f"[torch.compile] Triton not found - using eager backend (slower)", flush=True)
        except RuntimeError as e:
            print(f"[torch.compile] {e} Falling back to eager backend.", flush=True)
            backend = "eager"

        print(f"[torch.compile] Compiling HeartMuLa model (mode={compile_mode}, backend={backend})...", flush=True)
        print(f"[torch.compile] Note: First generation will be slower due to compilation.", flush=True)

        # Suppress errors and fall back to eager if compilation fails at runtime
        torch._dynamo.config.suppress_errors = True

        # Compile backbone and decoder
        model.backbone = torch.compile(
            model.backbone,
            backend=backend,
            mode=compile_mode,
            dynamic=True,
        )
        model.decoder = torch.compile(
            model.decoder,
            backend=backend,
            mode=compile_mode,
            dynamic=True,
        )

        print(f"[torch.compile] Model compiled successfully!", flush=True)
        return model

    except Exception as e:
        import warnings
        warnings.warn(f"torch.compile failed ({e}), continuing without compilation")
        print(f"[torch.compile] Compilation failed: {e}. Continuing without torch.compile.", flush=True)
        return model


def configure_flash_attention_for_gpu(device_id: int):
    """
    Configure Flash Attention based on GPU compute capability.
    Flash Attention / CUTLASS requires SM 7.0+ (Volta and newer).
    Older GPUs (Pascal, Maxwell, etc.) need the math backend.

    This is safe for all users:
    - CPU-only: Returns early, no CUDA settings modified
    - MPS (Apple Silicon): Returns early, not applicable
    - NVIDIA SM 7.0+: Enables Flash Attention for speed
    - NVIDIA SM 6.x and older: Disables Flash Attention, uses math backend
    - AMD ROCm: Conservatively disables Flash Attention (compatibility varies)
    """
    if not torch.cuda.is_available():
        logger.info("[GPU Config] CUDA not available - skipping Flash Attention configuration")
        return

    try:
        props = torch.cuda.get_device_properties(device_id)
        gpu_name = props.name.lower()
        compute_cap = props.major + props.minor / 10

        # Check for AMD ROCm GPUs (they report as CUDA but may not support Flash Attention)
        is_amd = any(x in gpu_name for x in ['gfx', 'amd', 'radeon', 'instinct'])

        if is_amd:
            # AMD GPUs: Conservatively disable Flash Attention
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            logger.info(f"[GPU Config] Flash Attention DISABLED for AMD GPU: {props.name} - using math backend")
            print(f"[GPU Config] Flash Attention DISABLED for AMD GPU: {props.name} - using math backend", flush=True)
        elif compute_cap >= 7.0:
            # NVIDIA SM 7.0+ (Volta, Turing, Ampere, Ada, Hopper) - enable Flash Attention
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info(f"[GPU Config] Flash Attention ENABLED for {props.name} (SM {props.major}.{props.minor})")
            print(f"[GPU Config] Flash Attention ENABLED for {props.name} (SM {props.major}.{props.minor})", flush=True)
        else:
            # Older NVIDIA GPUs (Pascal SM 6.x, Maxwell SM 5.x, etc.) - use math backend
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            logger.info(f"[GPU Config] Flash Attention DISABLED for {props.name} (SM {props.major}.{props.minor}) - using math backend")
            print(f"[GPU Config] Flash Attention DISABLED for {props.name} (SM {props.major}.{props.minor}) - using math backend", flush=True)

    except Exception as e:
        # If anything goes wrong, disable Flash Attention for safety
        logger.warning(f"[GPU Config] Error detecting GPU capabilities: {e}. Disabling Flash Attention for safety.")
        print(f"[GPU Config] Error detecting GPU: {e}. Disabling Flash Attention for safety.", flush=True)
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception:
            pass


def create_mmgp_pipeline(
    model_path: str,
    version: str,
    device: torch.device = None,
    optimization_level: str = "balanced",
    use_int8_quantization: bool = False,
):
    """
    Create a HeartMuLa pipeline using mmgp memory management.
    This is more reliable than BitsAndBytes on 12GB GPUs.

    Args:
        model_path: Path to model directory (where HuggingFace models are downloaded)
        version: Model version string
        device: CUDA device (default: cuda:0)
        optimization_level: mmgp profile ("conservative", "balanced", "aggressive")
        use_int8_quantization: Enable runtime INT8 quantization (slower but lower VRAM)

    Returns:
        HeartMuLaPipeline instance with mmgp profiling enabled
    """
    if not MMGP_AVAILABLE:
        raise RuntimeError("mmgp is not available. Install with: pip install mmgp accelerate")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[mmgp] Creating pipeline with {optimization_level} optimization...", flush=True)

    # NOTE: Pre-quantized INT8 models are NOT used for now.
    # When dequantized to bf16/f32 for inference, they're the same size as original models.
    # Pre-quantization only helps with disk storage, not VRAM. For 12GB GPUs, we still need
    # model swapping. Runtime INT8 quantization (quantizeTransformer=True) is an option
    # but is slower due to dequantization overhead on every forward pass.
    use_prequantized = False

    if use_prequantized:
        # This path is disabled - pre-quantized models don't save VRAM
        quantized_dir = os.path.join(model_path, "quantized")
        heartmula_weights_path = os.path.join(quantized_dir, "heartmula_int8.safetensors")
        print(f"[mmgp] Found pre-quantized INT8 models - using fast mode (no swapping)", flush=True)
    else:
        # Find the model weights path from HuggingFace download structure
        # The downloaded model is typically in a versioned folder like HeartMuLa-oss-RL-3B-20260123
        model_dir = os.path.join(model_path, f"HeartMuLa-oss-{version}")
        if not os.path.exists(model_dir):
            # Try without "oss" prefix
            model_dir = os.path.join(model_path, f"HeartMuLa-{version}")
        if not os.path.exists(model_dir):
            # Check for any HeartMuLa directory with the version suffix
            import glob
            candidates = glob.glob(os.path.join(model_path, f"*{version}*"))
            if candidates:
                model_dir = candidates[0]

        # Find the model weights (could be sharded or single file)
        # For sharded models, mmgp expects the first shard file, not the index.json
        import glob
        weights_single = os.path.join(model_dir, "model.safetensors")
        weights_shards = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))

        if weights_shards:
            # Sharded weights - pass the first shard to mmgp (it will find the others)
            heartmula_weights_path = weights_shards[0]
            print(f"[mmgp] Found sharded model weights ({len(weights_shards)} shards), using {heartmula_weights_path}", flush=True)
        elif os.path.exists(weights_single):
            heartmula_weights_path = weights_single
            print(f"[mmgp] Found single model weights at {weights_single}", flush=True)
        else:
            raise FileNotFoundError(f"Could not find HeartMuLa model weights in {model_dir}")

    # Create the pipeline with explicit weights path
    pipeline = HeartMuLaPipeline(
        ckpt_root=model_path,
        device=device,
        version=version,
        heartmula_weights_path=heartmula_weights_path,
    )

    # Set up mmgp profiling for memory management
    try:
        pipe_config = pipeline.get_mmgp_pipe_config(optimization_level)

        if use_prequantized:
            # Pre-quantized models fit in 12GB - use fast profile with no swapping
            profile_no = 1  # HighRAM_HighVRAM_Fastest
            pipeline._enable_async_codec_load = True  # Can preload codec
            # Allow models to coexist (both fit in VRAM)
            pipe_config["coTenantsMap"] = {
                "transformer": ["transformer2", "codec"],
                "transformer2": ["transformer", "codec"],
                "codec": ["transformer", "transformer2"],
            }
            print(f"[mmgp] Pre-quantized mode: both models fit in VRAM (no swapping)", flush=True)
        else:
            # bf16 models need swapping on 12GB
            profile_no = {"conservative": 1, "balanced": 3, "aggressive": 5}.get(optimization_level, 3)
            pipeline._enable_async_codec_load = False  # Can't preload - OOM

        print(f"[mmgp] Setting up profile {profile_no} for memory management...", flush=True)

        offload_obj = mmgp_offload.profile(
            pipe_config["pipe"],
            profile_no=profile_no,
            coTenantsMap=pipe_config.get("coTenantsMap"),
            budgets=pipe_config.get("budgets"),
            verboseLevel=0,
            # Runtime INT8 quantization: configurable via settings
            # bf16 is ~12% faster, but INT8 uses less VRAM
            quantizeTransformer=use_int8_quantization,
        )

        pipeline.set_offload_obj(offload_obj)
        print(f"[mmgp] Profiling enabled (profile {profile_no})", flush=True)

    except Exception as e:
        print(f"[mmgp] Warning: Profiling failed: {e}. Continuing without mmgp optimization.", flush=True)

    return pipeline


def create_quantized_pipeline(
    model_path: str,
    version: str,
    mula_device: torch.device,
    codec_device: torch.device,
    lazy_codec: bool = False,
    compile_model: bool = False,
    compile_mode: str = "default",
) -> HeartMuLaGenPipeline:
    """
    Create a HeartMuLa pipeline with 4-bit quantization for reduced VRAM usage.
    Uses BitsAndBytes NF4 quantization to reduce model size from ~11GB to ~3GB.

    Args:
        model_path: Path to model directory
        version: Model version string
        mula_device: Device for HeartMuLa model
        codec_device: Device for HeartCodec model
        lazy_codec: If True, don't load HeartCodec upfront - load only when needed for decoding.
                    This allows fitting on 12GB GPUs by never having both models in VRAM.
        compile_model: If True, apply torch.compile for faster inference
        compile_mode: torch.compile mode ("default", "reduce-overhead", "max-autotune")
    """
    from heartlib.pipelines.music_generation import _resolve_paths

    mula_path, codec_path, tokenizer_path, gen_config_path = _resolve_paths(model_path, version)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

    # Create 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[Quantization] Loading HeartMuLa with 4-bit NF4 quantization...", flush=True)

    # Load HeartMuLa with quantization
    heartmula = HeartMuLa.from_pretrained(
        mula_path,
        device_map=mula_device,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    
    # Apply torch.compile if enabled
    if compile_model:
        heartmula = apply_torch_compile(heartmula, compile_mode)

    heartcodec = None
    if not lazy_codec:
        # Load HeartCodec normally (it's smaller, no need for quantization)
        heartcodec = HeartCodec.from_pretrained(
            codec_path,
            device_map=codec_device,
            dtype=torch.float32,
        )
        print(f"[Quantization] Models loaded. HeartMuLa VRAM reduced by ~4x.", flush=True)
    else:
        print(f"[Quantization] HeartMuLa loaded (~3GB). HeartCodec will load lazily for decoding.", flush=True)

    # Create pipeline with lazy_load=True to avoid double loading
    # Then inject our pre-loaded quantized models
    pipeline = HeartMuLaGenPipeline(
        heartmula_path=mula_path,
        heartcodec_path=codec_path,
        heartmula_device=mula_device,
        heartcodec_device=codec_device,
        heartmula_dtype=torch.bfloat16,
        heartcodec_dtype=torch.float32,
        lazy_load=True,  # Don't load models in __init__
        muq_mulan=None,
        text_tokenizer=tokenizer,
        config=gen_config,
    )

    # Inject the pre-loaded quantized models
    pipeline._mula = heartmula
    pipeline._codec = heartcodec  # None if lazy_codec=True
    pipeline.lazy_load = lazy_codec  # Keep lazy mode if codec not loaded
    pipeline._lazy_codec = lazy_codec  # Track if we need lazy codec loading
    pipeline._codec_path = codec_path  # Store path for lazy loading

    return pipeline


def patch_pipeline_with_callback(pipeline: HeartMuLaGenPipeline, sequential_offload: bool = False):
    """
    Monkey-patch the HeartMuLa pipeline to support progress callbacks.
    This allows us to report generation progress without modifying upstream heartlib.

    We store a custom generate method on the pipeline instance that handles callbacks.

    Args:
        pipeline: The HeartMuLaGenPipeline to patch
        sequential_offload: If True, offload HeartMuLa to CPU after generation before loading
                           HeartCodec. This allows fitting on smaller GPUs like RTX 3060.
    """
    pipeline._sequential_offload = sequential_offload

    def generate_with_callback(inputs, callback=None, **kwargs):
        """Custom generate method that supports progress callback."""
        cfg_scale = kwargs.get("cfg_scale", 1.5)
        max_audio_length_ms = kwargs.get("max_audio_length_ms", 120_000)
        temperature = kwargs.get("temperature", 1.0)
        topk = kwargs.get("topk", 50)
        save_path = kwargs.get("save_path", "output.mp3")

        # Preprocess
        model_inputs = pipeline.preprocess(inputs, cfg_scale=cfg_scale)

        # Forward with progress callback
        prompt_tokens = model_inputs["tokens"].to(pipeline.mula_device)
        prompt_tokens_mask = model_inputs["tokens_mask"].to(pipeline.mula_device)
        continuous_segment = model_inputs["muq_embed"].to(pipeline.mula_device)
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"].to(pipeline.mula_device)
        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1
        pipeline.mula.setup_caches(bs_size)

        # Log VRAM usage after cache setup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"[VRAM] After cache setup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free", flush=True)

        with torch.autocast(device_type=pipeline.mula_device.type, dtype=pipeline.mula_dtype):
            curr_token = pipeline.mula.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
            )
        frames.append(curr_token[0:1,])

        def _pad_audio_token(token):
            padded_token = (
                torch.ones(
                    (token.shape[0], pipeline._parallel_number),
                    device=token.device,
                    dtype=torch.long,
                )
                * pipeline.config.empty_id
            )
            padded_token[:, :-1] = token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(
                padded_token, device=token.device, dtype=torch.bool
            )
            padded_token_mask[..., -1] = False
            return padded_token, padded_token_mask

        max_audio_frames = max_audio_length_ms // 80

        for i in tqdm(range(max_audio_frames), desc="Generating audio"):
            curr_token, curr_token_mask = _pad_audio_token(curr_token)
            with torch.autocast(device_type=pipeline.mula_device.type, dtype=pipeline.mula_dtype):
                curr_token = pipeline.mula.generate_frame(
                    tokens=curr_token,
                    tokens_mask=curr_token_mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,
                )
            if torch.any(curr_token[0:1, :] >= pipeline.config.audio_eos_id):
                break
            frames.append(curr_token[0:1,])

            # Report progress via callback
            if callback is not None:
                progress = int((i + 1) / max_audio_frames * 100)
                callback(progress, f"Generating audio... {i + 1}/{max_audio_frames} frames")

        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0).cpu()  # Move to CPU immediately

        # Sequential offload: Move HeartMuLa to CPU before loading HeartCodec
        # This allows fitting on smaller GPUs (12GB) by never having both models in VRAM
        if pipeline._sequential_offload:
            print("[Sequential Offload] Moving HeartMuLa to CPU...", flush=True)
            pipeline.mula.reset_caches()
            pipeline._mula.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[Sequential Offload] VRAM after offload: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)
        else:
            pipeline._unload()

        # Postprocess - load codec and decode
        if callback is not None:
            callback(95, "Decoding audio...")

        # Lazy codec loading: Load HeartCodec only when needed (for 12GB GPU mode)
        lazy_codec = getattr(pipeline, '_lazy_codec', False)
        if lazy_codec and pipeline._codec is None:
            print("[Lazy Loading] Loading HeartCodec for decoding...", flush=True)
            codec_path = getattr(pipeline, '_codec_path', None)
            if codec_path:
                pipeline._codec = HeartCodec.from_pretrained(
                    codec_path,
                    device_map=pipeline.codec_device,
                    dtype=torch.float32,
                )
                print(f"[Lazy Loading] HeartCodec loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)
            else:
                raise RuntimeError("Cannot load HeartCodec: codec_path not available")

        frames_for_codec = frames.to(pipeline.codec_device)
        wav = pipeline.codec.detokenize(frames_for_codec)

        # Cleanup codec if using lazy loading (free VRAM for next generation)
        if lazy_codec:
            print("[Lazy Loading] Unloading HeartCodec after decoding...", flush=True)
            del pipeline._codec
            pipeline._codec = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if pipeline._sequential_offload:
            # Move HeartMuLa back to GPU for next generation
            print("[Sequential Offload] Moving HeartMuLa back to GPU...", flush=True)
            pipeline._mula.to(pipeline.mula_device)
        elif not lazy_codec:
            pipeline._unload()

        # Save audio - try torchaudio first, fallback to soundfile for RTX 50 series (Blackwell)
        try:
            torchaudio.save(save_path, wav.to(torch.float32).cpu(), 48000)
        except Exception as e:
            # Fallback to soundfile for GPUs where torchcodec isn't supported (e.g., RTX 5080/5090)
            print(f"[Audio Save] torchaudio failed ({e}), using soundfile fallback...", flush=True)
            try:
                import soundfile as sf
                audio_np = wav.to(torch.float32).cpu().numpy().T  # (Channels, Time) -> (Time, Channels)
                sf.write(save_path, audio_np, 48000)
                print(f"[Audio Save] Saved with soundfile successfully", flush=True)
            except ImportError:
                raise RuntimeError("Audio save failed. Please install soundfile: pip install soundfile")

    # Store the custom method on the pipeline instance
    pipeline.generate_with_callback = generate_with_callback

    logger.info("[Pipeline] Patched HeartMuLa pipeline with callback support")
    lazy_codec = getattr(pipeline, '_lazy_codec', False)
    print(f"[Pipeline] Patched with callback support (sequential_offload={sequential_offload}, lazy_codec={lazy_codec})", flush=True)
    return pipeline


def cleanup_gpu_memory():
    """Clean up GPU memory before loading models."""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        logger.info("GPU memory cleaned up")


def get_gpu_memory(device_id):
    props = torch.cuda.get_device_properties(device_id)
    return props.total_memory / (1024 ** 3)


class MusicService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MusicService, cls).__new__(cls)
            cls._instance.pipeline = None
            cls._instance.gpu_lock = asyncio.Lock()
            cls._instance.is_loading = False
            cls._instance.active_jobs = {}  # Map job_id -> threading.Event
            cls._instance.gpu_mode = "single"
            cls._instance.job_queue = []  # List of job_ids waiting in queue
            # Startup progress tracking
            cls._instance.startup_status = "not_started"  # not_started, downloading, loading, ready, error
            cls._instance.startup_progress = 0
            cls._instance.startup_message = ""
            cls._instance.startup_error = None
            # torch.compile first run tracking
            cls._instance._torch_compile_first_run = True
            # CUDA error recovery flag
            cls._instance._cuda_error_occurred = False
            # mmgp mode flag (set when pipeline is loaded)
            cls._instance._using_mmgp = False
            # Current settings (for display in settings panel)
            cls._instance.current_settings = {
                "quantization_4bit": "auto",
                "sequential_offload": "auto",
                "torch_compile": False,
                "torch_compile_mode": "default",
                # mmgp-specific settings
                "mmgp_quantization": "false",  # "true" for INT8, "false" for bf16 (faster)
                # LLM Provider settings
                "ollama_host": "",
                "openrouter_api_key": "",
                # Custom API settings
                "custom_api_base_url": "",
                "custom_api_key": "",
                "custom_api_model": ""
            }
            # Load persisted settings from disk (overrides defaults)
            cls._instance._load_settings()
        return cls._instance

    def _load_settings(self):
        """Load settings from persistent storage and apply to global variables."""
        global ENABLE_4BIT_QUANTIZATION, ENABLE_SEQUENTIAL_OFFLOAD, ENABLE_TORCH_COMPILE, TORCH_COMPILE_MODE
        global _4BIT_ENV, _OFFLOAD_ENV, _COMPILE_ENV, _COMPILE_MODE_ENV

        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    saved = json.load(f)
                    # Only update keys that exist in defaults
                    for key in self.current_settings:
                        if key in saved:
                            self.current_settings[key] = saved[key]

                    # Apply loaded settings to global variables
                    if "quantization_4bit" in saved:
                        val = saved["quantization_4bit"]
                        _4BIT_ENV = val
                        if val == "auto":
                            ENABLE_4BIT_QUANTIZATION = None
                        else:
                            ENABLE_4BIT_QUANTIZATION = val == "true"

                    if "sequential_offload" in saved:
                        val = saved["sequential_offload"]
                        _OFFLOAD_ENV = val
                        if val == "auto":
                            ENABLE_SEQUENTIAL_OFFLOAD = None
                        else:
                            ENABLE_SEQUENTIAL_OFFLOAD = val == "true"

                    if "torch_compile" in saved:
                        ENABLE_TORCH_COMPILE = saved["torch_compile"]
                        _COMPILE_ENV = "true" if saved["torch_compile"] else "false"

                    if "torch_compile_mode" in saved:
                        TORCH_COMPILE_MODE = saved["torch_compile_mode"]
                        _COMPILE_MODE_ENV = saved["torch_compile_mode"]

                    # Apply LLM settings
                    from backend.app.services.llm_service import LLMService
                    if "ollama_host" in saved and saved["ollama_host"]:
                        LLMService.update_settings(ollama_host=saved["ollama_host"])
                    if "openrouter_api_key" in saved and saved["openrouter_api_key"]:
                        LLMService.update_settings(openrouter_api_key=saved["openrouter_api_key"])
                    # Apply Custom API settings
                    if "custom_api_base_url" in saved:
                        LLMService.update_settings(custom_api_base_url=saved["custom_api_base_url"])
                    if "custom_api_key" in saved:
                        LLMService.update_settings(custom_api_key=saved["custom_api_key"])
                    if "custom_api_model" in saved:
                        LLMService.update_settings(custom_api_model=saved["custom_api_model"])

                    logger.info(f"[Settings] Loaded from {SETTINGS_FILE}: {self.current_settings}")
        except Exception as e:
            logger.warning(f"[Settings] Failed to load settings: {e}")

    def _save_settings(self):
        """Save settings to persistent storage."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.current_settings, f, indent=2)
            logger.info(f"[Settings] Saved to {SETTINGS_FILE}")
        except Exception as e:
            logger.warning(f"[Settings] Failed to save settings: {e}")

    def get_queue_position(self, job_id: str) -> int:
        """Returns 1-based position in queue, or 0 if not in queue."""
        try:
            return self.job_queue.index(job_id) + 1
        except ValueError:
            return 0

    def _broadcast_queue_update(self):
        """Notify all queued jobs of their current position."""
        for i, jid in enumerate(self.job_queue):
            event_manager.publish("job_queue", {"job_id": str(jid), "position": i + 1, "total": len(self.job_queue)})

    def _emit_startup_progress(self, status: str, progress: int, message: str, error: str = None):
        """Emit startup progress via SSE."""
        self.startup_status = status
        self.startup_progress = progress
        self.startup_message = message
        self.startup_error = error
        event_manager.publish("startup_progress", {
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "ready": status == "ready"
        })
        print(f"[Startup] {status}: {progress}% - {message}", flush=True)

    def get_startup_status(self) -> dict:
        """Get current startup status."""
        return {
            "status": self.startup_status,
            "progress": self.startup_progress,
            "message": self.startup_message,
            "error": self.startup_error,
            "ready": self.startup_status == "ready"
        }

    async def _download_models_with_progress(self, model_dir: str, version: str) -> str:
        """Download models with progress callbacks."""
        from huggingface_hub import snapshot_download, hf_hub_download

        os.makedirs(model_dir, exist_ok=True)

        # Get HuggingFace repo ID and local folder name for this version
        if version in MODEL_VERSIONS:
            hf_repo, folder_name = MODEL_VERSIONS[version]
        else:
            hf_repo = f"HeartMuLa/HeartMuLa-oss-{version}"
            folder_name = f"HeartMuLa-oss-{version}"

        heartmula_path = os.path.join(model_dir, folder_name)
        heartcodec_path = os.path.join(model_dir, "HeartCodec-oss")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        gen_config_path = os.path.join(model_dir, "gen_config.json")

        all_present = (
            os.path.exists(heartmula_path) and
            os.path.exists(heartcodec_path) and
            os.path.isfile(tokenizer_path) and
            os.path.isfile(gen_config_path)
        )

        if all_present:
            self._emit_startup_progress("downloading", 40, "All models found locally")
            return model_dir

        loop = asyncio.get_running_loop()

        # Download HeartMuLa model
        if not os.path.exists(heartmula_path):
            self._emit_startup_progress("downloading", 10, f"Downloading {folder_name} (~3GB)...")
            await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=hf_repo,
                    local_dir=heartmula_path,
                    local_dir_use_symlinks=False,
                )
            )
            self._emit_startup_progress("downloading", 28, f"{folder_name} downloaded")

        # Download HeartCodec model
        if not os.path.exists(heartcodec_path):
            self._emit_startup_progress("downloading", 30, "Downloading HeartCodec (~1.5GB)...")
            await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=HF_HEARTCODEC_REPO,
                    local_dir=heartcodec_path,
                    local_dir_use_symlinks=False,
                )
            )
            self._emit_startup_progress("downloading", 38, "HeartCodec downloaded")

        # Download tokenizer and gen_config
        if not os.path.isfile(tokenizer_path):
            self._emit_startup_progress("downloading", 39, "Downloading tokenizer...")
            await loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=HF_HEARTMULA_GEN_REPO,
                    filename="tokenizer.json",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                )
            )

        if not os.path.isfile(gen_config_path):
            await loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=HF_HEARTMULA_GEN_REPO,
                    filename="gen_config.json",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                )
            )

        self._emit_startup_progress("downloading", 40, "All models downloaded")
        return model_dir

    async def initialize_with_progress(self, model_path: Optional[str] = None, version: str = None):
        """Initialize models with progress events for frontend display."""
        if self.pipeline is not None or self.is_loading:
            if self.pipeline is not None:
                self._emit_startup_progress("ready", 100, "Ready!")
            return

        self.is_loading = True

        try:
            # Use default version if not specified
            if version is None:
                version = DEFAULT_VERSION

            self._emit_startup_progress("downloading", 0, "Checking models...")

            # Clean up GPU memory
            self._emit_startup_progress("downloading", 5, "Preparing GPU...")
            cleanup_gpu_memory()

            # Download models with progress
            if model_path is None:
                model_path = await self._download_models_with_progress(DEFAULT_MODEL_DIR, version)

            # Load models
            self._emit_startup_progress("loading", 45, "Loading HeartMuLa model...")

            loop = asyncio.get_running_loop()

            # Store settings being used (preserve LLM settings)
            self.current_settings.update({
                "quantization_4bit": _4BIT_ENV,
                "sequential_offload": _OFFLOAD_ENV,
                "torch_compile": ENABLE_TORCH_COMPILE,
                "torch_compile_mode": TORCH_COMPILE_MODE
            })

            self.pipeline = await loop.run_in_executor(
                None,
                lambda mp=model_path, v=version: self._load_pipeline_multi_gpu(mp, v)
            )

            self._emit_startup_progress("loading", 95, "Initializing pipeline...")

            self._emit_startup_progress("ready", 100, "Ready!")
            logger.info(f"Heartlib model loaded successfully in {self.gpu_mode}-GPU mode.")

            # Save settings to disk after successful initialization
            self._save_settings()

        except Exception as e:
            logger.error(f"Failed to load Heartlib model: {e}")
            self._emit_startup_progress("error", 0, f"Failed to load model: {str(e)}", str(e))
            raise e
        finally:
            self.is_loading = False

    def _unload_all_models(self):
        """Unload all models and free GPU memory."""
        logger.info("Unloading all models...")
        if self.pipeline is not None:
            # Unload HeartMuLa
            if hasattr(self.pipeline, '_mula') and self.pipeline._mula is not None:
                del self.pipeline._mula
            # Unload HeartCodec
            if hasattr(self.pipeline, '_codec') and self.pipeline._codec is not None:
                del self.pipeline._codec
            del self.pipeline
            self.pipeline = None

        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        logger.info("All models unloaded")

    def get_gpu_info(self) -> dict:
        """Get GPU hardware information."""
        result = {
            "cuda_available": torch.cuda.is_available(),
            "num_gpus": 0,
            "gpus": [],
            "total_vram_gb": 0
        }

        if not torch.cuda.is_available():
            return result

        num_gpus = torch.cuda.device_count()
        result["num_gpus"] = num_gpus

        total_vram = 0
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            compute_cap = props.major + props.minor / 10
            gpu_info = {
                "index": i,
                "name": props.name,
                "vram_gb": round(vram_gb, 1),
                "compute_capability": compute_cap,
                "supports_flash_attention": compute_cap >= 7.0
            }
            result["gpus"].append(gpu_info)
            total_vram += vram_gb

        result["total_vram_gb"] = round(total_vram, 1)
        return result

    async def reload_models(self, settings: dict):
        """Reload models with new settings."""
        # Check if a job is currently processing
        if len(self.active_jobs) > 0:
            raise RuntimeError("Cannot reload models while a job is processing")

        if len(self.job_queue) > 0:
            raise RuntimeError("Cannot reload models while jobs are queued")

        logger.info(f"Reloading models with settings: {settings}")

        # Emit reload started event
        self._emit_startup_progress("loading", 0, "Reloading models...")

        # Apply new settings as environment variables (for next load)
        global ENABLE_4BIT_QUANTIZATION, ENABLE_SEQUENTIAL_OFFLOAD, ENABLE_TORCH_COMPILE, TORCH_COMPILE_MODE
        global _4BIT_ENV, _OFFLOAD_ENV, _COMPILE_ENV, _COMPILE_MODE_ENV

        if "quantization_4bit" in settings:
            val = settings["quantization_4bit"]
            _4BIT_ENV = val
            if val == "auto":
                ENABLE_4BIT_QUANTIZATION = None
            else:
                ENABLE_4BIT_QUANTIZATION = val == "true"

        if "sequential_offload" in settings:
            val = settings["sequential_offload"]
            _OFFLOAD_ENV = val
            if val == "auto":
                ENABLE_SEQUENTIAL_OFFLOAD = None
            else:
                ENABLE_SEQUENTIAL_OFFLOAD = val == "true"

        if "torch_compile" in settings:
            ENABLE_TORCH_COMPILE = settings["torch_compile"]
            _COMPILE_ENV = "true" if settings["torch_compile"] else "false"

        if "torch_compile_mode" in settings:
            TORCH_COMPILE_MODE = settings["torch_compile_mode"]
            _COMPILE_MODE_ENV = settings["torch_compile_mode"]

        # Reset first run flag when torch.compile setting changes
        if "torch_compile" in settings:
            self._torch_compile_first_run = settings["torch_compile"]

        # Unload current models
        self._emit_startup_progress("loading", 10, "Unloading current models...")
        self._unload_all_models()

        # Reinitialize with new settings
        await self.initialize_with_progress()

    def _load_pipeline_multi_gpu(self, model_path: str, version: str):
        """Load pipeline with multi-GPU support and automatic VRAM-based configuration."""

        # Auto-detect optimal configuration if not manually specified
        auto_config = detect_optimal_gpu_config()
        num_gpus = auto_config["num_gpus"]

        # Check if mmgp should be used (preferred for single GPU 12GB setups)
        use_mmgp = USE_MMGP
        if use_mmgp is None:
            # Auto-detect: use mmgp for single GPU with < 20GB VRAM when available
            if MMGP_AVAILABLE and num_gpus == 1:
                gpu_info = auto_config.get("gpu_info", {})
                if gpu_info:
                    vram = list(gpu_info.values())[0].get("vram_gb", 0)
                    use_mmgp = vram < 20  # Use mmgp for < 20GB GPUs
                else:
                    use_mmgp = True  # Default to mmgp for single GPU
            else:
                use_mmgp = False

        if use_mmgp and MMGP_AVAILABLE:
            print(f"[Config] Using mmgp memory management (recommended for this GPU)", flush=True)
            self.gpu_mode = "mmgp"
            self.gpu_config = auto_config

            # Configure Flash Attention
            configure_flash_attention_for_gpu(0)

            # Determine optimization level based on VRAM
            gpu_info = auto_config.get("gpu_info", {})
            vram = list(gpu_info.values())[0].get("vram_gb", 12) if gpu_info else 12

            if vram < 10:
                opt_level = "conservative"
            elif vram < 16:
                opt_level = "balanced"
            else:
                opt_level = "aggressive"

            # Check if INT8 quantization is enabled in settings
            use_int8 = self.current_settings.get("mmgp_quantization", "false") == "true"
            if use_int8:
                print(f"[mmgp] INT8 quantization ENABLED (from settings)", flush=True)
            else:
                print(f"[mmgp] Using bf16 precision (faster)", flush=True)

            pipeline = create_mmgp_pipeline(
                model_path, version,
                device=torch.device("cuda:0"),
                optimization_level=opt_level,
                use_int8_quantization=use_int8,
            )

            # Store that we're using mmgp mode
            self._using_mmgp = True
            return pipeline

        # Fall back to heartlib/BitsAndBytes mode
        self._using_mmgp = False

        # Use manual override if set, otherwise use auto-detected values
        if ENABLE_4BIT_QUANTIZATION is not None:
            use_quantization = ENABLE_4BIT_QUANTIZATION and QUANTIZATION_AVAILABLE
            print(f"[Config] Using manually set HEARTMULA_4BIT={ENABLE_4BIT_QUANTIZATION}", flush=True)
        else:
            use_quantization = auto_config["use_quantization"] and QUANTIZATION_AVAILABLE
            print(f"[Config] Auto-detected: 4-bit quantization = {use_quantization}", flush=True)

        if ENABLE_SEQUENTIAL_OFFLOAD is not None:
            use_sequential_offload = ENABLE_SEQUENTIAL_OFFLOAD
            print(f"[Config] Using manually set HEARTMULA_SEQUENTIAL_OFFLOAD={ENABLE_SEQUENTIAL_OFFLOAD}", flush=True)
        else:
            use_sequential_offload = auto_config["use_sequential_offload"]
            print(f"[Config] Auto-detected: sequential offload = {use_sequential_offload}", flush=True)

        # torch.compile settings
        use_compile = ENABLE_TORCH_COMPILE
        compile_mode = TORCH_COMPILE_MODE
        if use_compile:
            print(f"[Config] torch.compile ENABLED (mode={compile_mode})", flush=True)
        else:
            print(f"[Config] torch.compile DISABLED", flush=True)

        # Store the detected config for reference
        self.gpu_config = auto_config

        if use_quantization:
            print(f"[Quantization] 4-bit quantization ENABLED - model will use ~3GB instead of ~11GB", flush=True)
        else:
            print(f"[Quantization] 4-bit quantization DISABLED - using full precision (~11GB)", flush=True)

        if num_gpus < 2:
            logger.info(f"Found {num_gpus} GPU(s). Using single GPU mode...")
            self.gpu_mode = "single"

            # Configure Flash Attention for the GPU
            configure_flash_attention_for_gpu(0)

            if use_quantization:
                if use_sequential_offload:
                    # 12GB GPU mode: Load only HeartMuLa upfront, lazy load HeartCodec
                    # This allows: HeartMuLa 4-bit (~3GB) + KV cache (~4GB) = ~7GB during generation
                    # Then swap: unload HeartMuLa, load HeartCodec (~6GB) for decoding
                    print("[12GB GPU Mode] Using lazy codec loading for 12GB GPU", flush=True)
                    pipeline = create_quantized_pipeline(
                        model_path, version,
                        mula_device=torch.device("cuda"),
                        codec_device=torch.device("cuda"),
                        lazy_codec=True,  # Don't load HeartCodec upfront
                        compile_model=use_compile,
                        compile_mode=compile_mode,
                    )
                    return patch_pipeline_with_callback(pipeline, sequential_offload=True)
                else:
                    # With quantization on larger VRAM GPU, model fits easily - use GPU for both
                    print("[14GB+ GPU Mode] Both models fit in VRAM without swapping", flush=True)
                    pipeline = create_quantized_pipeline(
                        model_path, version,
                        mula_device=torch.device("cuda"),
                        codec_device=torch.device("cuda"),
                        lazy_codec=False,
                        compile_model=use_compile,
                        compile_mode=compile_mode,
                    )
                    return patch_pipeline_with_callback(pipeline, sequential_offload=False)
            elif use_sequential_offload:
                # Sequential offload mode for 12GB GPUs without quantization
                # Both models on same GPU but loaded/unloaded sequentially
                print("[Sequential Offload] ENABLED - models will be swapped between GPU and CPU", flush=True)
                pipeline = HeartMuLaGenPipeline.from_pretrained(
                    model_path,
                    device={
                        "mula": torch.device("cuda"),
                        "codec": torch.device("cuda"),  # Will be loaded when needed
                    },
                    dtype={
                        "mula": torch.bfloat16,
                        "codec": torch.float32,
                    },
                    version=version,
                    lazy_load=True,
                )
                # Apply torch.compile if enabled
                if use_compile:
                    pipeline._mula = apply_torch_compile(pipeline._mula, compile_mode)
                return patch_pipeline_with_callback(pipeline, sequential_offload=True)
            else:
                # Without quantization, use lazy loading - codec stays on CPU
                pipeline = HeartMuLaGenPipeline.from_pretrained(
                    model_path,
                    device={
                        "mula": torch.device("cuda"),
                        "codec": torch.device("cpu"),
                    },
                    dtype={
                        "mula": torch.bfloat16,
                        "codec": torch.float32,
                    },
                    version=version,
                    lazy_load=True,
                )
                # Apply torch.compile if enabled
                if use_compile:
                    pipeline._mula = apply_torch_compile(pipeline._mula, compile_mode)
                return patch_pipeline_with_callback(pipeline, sequential_offload=False)

        # Multi-GPU setup
        logger.info(f"Found {num_gpus} GPUs:")
        gpu_info = {}
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            mem = props.total_memory / (1024 ** 3)
            compute_cap = props.major + props.minor / 10
            gpu_info[i] = {"mem": mem, "compute": compute_cap, "name": props.name}
            logger.info(f"  GPU {i}: {props.name} ({mem:.1f} GB, SM {props.major}.{props.minor})")
            print(f"[GPU Setup] GPU {i}: {props.name} ({mem:.1f} GB, SM {props.major}.{props.minor})", flush=True)

        if use_quantization:
            # With 4-bit quantization: HeartMuLa only needs ~3GB
            # Prioritize compute capability (faster GPU with Flash Attention) for HeartMuLa
            # Put HeartCodec (~6GB) on the GPU with more VRAM
            mula_gpu = max(gpu_info, key=lambda x: gpu_info[x]["compute"])
            codec_gpu = max(gpu_info, key=lambda x: gpu_info[x]["mem"])
            # If same GPU has both best compute and most VRAM, use the other for codec
            if mula_gpu == codec_gpu and num_gpus > 1:
                codec_gpu = min(gpu_info, key=lambda x: gpu_info[x]["compute"])
            print(f"[GPU Setup] 4-bit mode: HeartMuLa on fastest GPU, HeartCodec on largest VRAM GPU", flush=True)
        else:
            # Without quantization: HeartMuLa needs ~11GB, prioritize VRAM
            mula_gpu = max(gpu_info, key=lambda x: gpu_info[x]["mem"])
            codec_gpu = min(gpu_info, key=lambda x: gpu_info[x]["mem"])
            print(f"[GPU Setup] Full precision: HeartMuLa on largest VRAM GPU", flush=True)

        print(f"[GPU Setup] HeartMuLa -> GPU {mula_gpu}: {gpu_info[mula_gpu]['name']} ({gpu_info[mula_gpu]['mem']:.1f} GB, SM {gpu_info[mula_gpu]['compute']})", flush=True)
        print(f"[GPU Setup] HeartCodec -> GPU {codec_gpu}: {gpu_info[codec_gpu]['name']} ({gpu_info[codec_gpu]['mem']:.1f} GB)", flush=True)

        # Configure Flash Attention based on the GPU running HeartMuLa
        configure_flash_attention_for_gpu(mula_gpu)

        self.gpu_mode = "multi"

        if use_quantization:
            pipeline = create_quantized_pipeline(
                model_path, version,
                mula_device=torch.device(f"cuda:{mula_gpu}"),
                codec_device=torch.device(f"cuda:{codec_gpu}"),
                compile_model=use_compile,
                compile_mode=compile_mode,
            )
        else:
            pipeline = HeartMuLaGenPipeline.from_pretrained(
                model_path,
                device={
                    "mula": torch.device(f"cuda:{mula_gpu}"),
                    "codec": torch.device(f"cuda:{codec_gpu}"),
                },
                dtype={
                    "mula": torch.bfloat16,
                    "codec": torch.float32,
                },
                version=version,
            )
            # Apply torch.compile if enabled
            if use_compile:
                pipeline._mula = apply_torch_compile(pipeline._mula, compile_mode)
        return patch_pipeline_with_callback(pipeline, sequential_offload=False)

    async def initialize(self, model_path: Optional[str] = None, version: str = None):
        if self.pipeline is not None or self.is_loading:
            return

        self.is_loading = True

        # Use default version if not specified
        if version is None:
            version = DEFAULT_VERSION
        logger.info(f"Using HeartMuLa version: {version}")

        # Clean up GPU memory before loading
        logger.info("Cleaning up GPU memory before loading...")
        cleanup_gpu_memory()

        # Auto-download models if not present
        loop = asyncio.get_running_loop()
        if model_path is None:
            logger.info("Checking for models and downloading if needed...")
            model_path = await loop.run_in_executor(
                None,
                lambda v=version: ensure_models_downloaded(DEFAULT_MODEL_DIR, v)
            )

        logger.info(f"Loading Heartlib model from {model_path}...")
        try:
            # Run blocking load in executor to avoid freezing async loop
            self.pipeline = await loop.run_in_executor(
                None,
                lambda mp=model_path, v=version: self._load_pipeline_multi_gpu(mp, v)
            )
            logger.info(f"Heartlib model loaded successfully in {self.gpu_mode}-GPU mode.")
        except Exception as e:
            logger.error(f"Failed to load Heartlib model: {e}")
            raise e
        finally:
            self.is_loading = False

    async def generate_task(self, job_id: str, request: GenerationRequest, db_engine):
        """Background task to generate music."""
        from uuid import UUID as PyUUID
        job_id_str = str(job_id)  # String for dictionary keys
        job_id_uuid = PyUUID(job_id_str) if isinstance(job_id, str) else job_id  # UUID for DB queries

        # Add to queue and broadcast position
        self.job_queue.append(job_id_str)
        queue_pos = self.get_queue_position(job_id_str)
        logger.info(f"Job {job_id_str} added to queue at position {queue_pos}")
        event_manager.publish("job_queued", {"job_id": job_id_str, "position": queue_pos, "total": len(self.job_queue)})
        self._broadcast_queue_update()

        # 1. Acquire GPU Lock (will wait if another job is processing)
        async with self.gpu_lock:
            # Remove from queue now that we have the lock
            if job_id_str in self.job_queue:
                self.job_queue.remove(job_id_str)
                self._broadcast_queue_update()  # Update remaining jobs' positions

            logger.info(f"Starting generation for job {job_id_str}")

            # Check for CUDA error recovery
            if getattr(self, '_cuda_error_occurred', False):
                logger.info("[CUDA Recovery] Previous CUDA error detected. Reloading models...")
                print("[CUDA Recovery] Reloading models after previous CUDA error...", flush=True)
                event_manager.publish("job_progress", {"job_id": job_id_str, "progress": 0, "msg": "Recovering from GPU error..."})
                try:
                    self._unload_all_models()
                    # Reset CUDA context
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # Reset all GPU devices
                        for i in range(torch.cuda.device_count()):
                            with torch.cuda.device(i):
                                torch.cuda.reset_peak_memory_stats()
                    # Reload models
                    await self.initialize_with_progress()
                    self._cuda_error_occurred = False
                    logger.info("[CUDA Recovery] Models reloaded successfully")
                    print("[CUDA Recovery] Models reloaded successfully", flush=True)
                except Exception as recovery_err:
                    logger.error(f"[CUDA Recovery] Failed to recover: {recovery_err}")
                    print(f"[CUDA Recovery] Failed to recover: {recovery_err}", flush=True)
                    # Continue anyway - the next generation might still work

            # 2. Update status to PROCESSING
            import time
            generation_start_time = time.time()

            try:
                with Session(db_engine) as session:
                    # check if job still exists
                    job = session.exec(select(Job).where(Job.id == job_id_uuid)).one_or_none()
                    if not job:
                        logger.warning(f"Job {job_id_str} was deleted before processing started. Aborting.")
                        return

                    job.status = JobStatus.PROCESSING
                    session.add(job)
                    session.commit()
                    logger.info(f"Job {job_id_str} status updated to PROCESSING")
            except Exception as e:
                logger.error(f"Failed to update job status to PROCESSING: {e}")
                return

            try:
                # 3. Create unique filename
                output_filename = f"song_{job_id_str}.mp3"
                save_path = os.path.abspath(f"backend/generated_audio/{output_filename}")

                # Create Cancellation Event
                import threading
                abort_event = threading.Event()
                self.active_jobs[job_id_str] = abort_event

                # 4. Generate Title (User-provided > Lyrics extraction > LLM)
                from backend.app.services.llm_service import LLMService
                import re

                auto_title = "Untitled Track"

                # Priority 1: User-provided title
                if request.title and request.title.strip():
                    auto_title = request.title.strip()
                    logger.info(f"Using user-provided title: {auto_title}")
                else:
                    # Priority 2: Extract from lyrics (first meaningful line, excluding markers)
                    if request.lyrics and len(request.lyrics) > 5:
                        # Split lyrics into lines and find first non-marker line
                        lines = request.lyrics.strip().split('\n')
                        marker_pattern = re.compile(r'^\s*\[.*\]\s*$|^\s*\(.*\)\s*$', re.IGNORECASE)
                        section_words = {'intro', 'verse', 'chorus', 'bridge', 'outro', 'hook', 'pre-chorus', 'interlude', 'break'}

                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            # Skip lines that are just markers like [Intro], [Verse 1], (Chorus), etc.
                            if marker_pattern.match(line):
                                continue
                            # Skip lines that are just section words
                            if line.lower().replace('[', '').replace(']', '').strip() in section_words:
                                continue
                            # Found a meaningful line - use first 50 chars as title
                            extracted_title = line[:50].strip()
                            if len(line) > 50:
                                extracted_title = extracted_title.rsplit(' ', 1)[0] + '...'
                            auto_title = extracted_title
                            logger.info(f"Extracted title from lyrics: {auto_title}")
                            break

                    # Priority 3: LLM generation (only if we still have default title)
                    if auto_title == "Untitled Track":
                        try:
                            # Use lyrics for context if available, otherwise prompt
                            context_source = request.lyrics if request.lyrics and len(request.lyrics) > 10 else request.prompt
                            # Truncate to first 1000 chars to avoid token limits, but enough for context
                            context_source = context_source[:1000]

                            # Logic: If no model is specified, find what's running locally
                            model_to_use = request.llm_model
                            provider_to_use = "ollama"
                            if not model_to_use:
                                try:
                                    models = LLMService.get_models()
                                    if models:
                                        model_to_use = models[0]["id"]
                                        provider_to_use = models[0]["provider"]
                                        logger.info(f"No specific LLM model requested. Using: {model_to_use} ({provider_to_use})")
                                    else:
                                        model_to_use = "llama3"
                                        logger.warning("No specific LLM model requested and no local models found. Defaulting to 'llama3'.")
                                except Exception as e:
                                    model_to_use = "llama3"
                                    logger.warning(f"Error fetching local models: {e}. Fallback to 'llama3'.")

                            auto_title = LLMService.generate_title(context_source, model=model_to_use, provider=provider_to_use)
                        except Exception as e:
                            logger.warning(f"Auto-title generation failed: {e}. Using default.")

                # 5. Run Generation (Blocking, run in executor)

                # Set Seed
                seed_to_use = request.seed
                if seed_to_use is None:
                    import random
                    seed_to_use = random.randint(0, 2**32 - 1)

                # Note: heartlib's pipeline is not async, so we wrap it
                loop = asyncio.get_running_loop()

                # Progress Callback for Pipeline
                def _pipeline_callback(progress, msg):
                    import warnings
                    warnings.filterwarnings("ignore", message="In MPS autocast, but the target dtype is not supported")

                    loop.call_soon_threadsafe(
                        event_manager.publish,
                        "job_progress",
                        {"job_id": job_id_str, "progress": progress, "msg": msg}
                    )

                def _run_pipeline():
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

                    sound_tags = request.tags if request.tags and request.tags.strip() else "pop music"

                    logger.info(f"Setting random seed to {seed_to_use}")
                    torch.manual_seed(seed_to_use)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_to_use)
                    import random
                    random.seed(seed_to_use)
                    import numpy as np
                    np.random.seed(seed_to_use)

                    # Resolve reference audio path if provided
                    ref_audio_path = None
                    muq_segment_sec = 60.0
                    print(f"[DEBUG] Ref audio ID from request: {request.ref_audio_id}", flush=True)
                    if request.ref_audio_id:
                        ref_audio_dir = os.path.join(os.getcwd(), "backend", "ref_audio")
                        for ext in [".mp3", ".wav", ".flac", ".ogg"]:
                            candidate = os.path.join(ref_audio_dir, f"{request.ref_audio_id}{ext}")
                            if os.path.exists(candidate):
                                ref_audio_path = candidate
                                logger.info(f"Using reference audio: {ref_audio_path}")
                                try:
                                    info = torchaudio.info(ref_audio_path)
                                    audio_duration_sec = info.num_frames / info.sample_rate
                                    max_segment_sec = 10.0
                                    muq_segment_sec = (request.style_influence / 100.0) * max_segment_sec
                                    muq_segment_sec = min(muq_segment_sec, audio_duration_sec)
                                    muq_segment_sec = max(1.0, muq_segment_sec)
                                    print(f"[DEBUG] Audio duration: {audio_duration_sec:.1f}s, style_influence: {request.style_influence}%, muq_segment_sec: {muq_segment_sec:.1f}s", flush=True)
                                except Exception as e:
                                    logger.warning(f"Could not get audio duration: {e}, using default 10s")
                                    muq_segment_sec = 10.0
                                break

                    with torch.no_grad():
                        # Check if using mmgp pipeline (different interface)
                        if getattr(self, '_using_mmgp', False):
                            # mmgp pipeline has different interface:
                            # - input_prompt = lyrics
                            # - alt_prompt = tags
                            # - Returns dict with 'x' (audio tensor) and 'audio_sampling_rate'
                            print(f"[mmgp] Generating with mmgp pipeline...", flush=True)

                            lyrics_text = request.lyrics or ""
                            if not lyrics_text.strip():
                                # For instrumental mode, use "[instrumental]" as lyrics
                                lyrics_text = "[instrumental]"

                            if ref_audio_path:
                                print(f"[mmgp] Warning: Reference audio not yet supported in mmgp mode", flush=True)

                            # Adapter callback: mmgp uses step_idx/override_num_inference_steps
                            # but our callback expects (progress, msg)
                            def _mmgp_callback_adapter(step_idx=-1, override_num_inference_steps=None, force_refresh=False, **kwargs):
                                if override_num_inference_steps and override_num_inference_steps > 0:
                                    if step_idx >= 0:
                                        progress = int((step_idx + 1) / override_num_inference_steps * 100)
                                        generated_sec = step_idx + 1
                                        msg = f"Generating... {generated_sec}s / {override_num_inference_steps}s"
                                    else:
                                        progress = 0
                                        msg = "Starting generation..."
                                elif force_refresh:
                                    progress = 95
                                    msg = "Decoding audio..."
                                else:
                                    progress = 0
                                    msg = "Processing..."
                                _pipeline_callback(progress, msg)

                            result = self.pipeline.generate(
                                input_prompt=lyrics_text,
                                model_mode=None,  # Not used
                                audio_guide=None,  # Reference audio not supported yet
                                alt_prompt=sound_tags,
                                temperature=request.temperature,
                                max_audio_length_ms=request.duration_ms,
                                topk=request.topk,
                                cfg_scale=request.cfg_scale,
                                seed=seed_to_use,
                                callback=_mmgp_callback_adapter,
                            )

                            if result is not None and "x" in result:
                                # Save the audio tensor to file
                                wav = result["x"]
                                sample_rate = result.get("audio_sampling_rate", 48000)

                                # Ensure tensor is on CPU and in correct format
                                if isinstance(wav, torch.Tensor):
                                    wav = wav.cpu()
                                    if wav.dim() == 1:
                                        wav = wav.unsqueeze(0)  # Add channel dimension

                                # Save using torchaudio (preferred) or soundfile
                                try:
                                    torchaudio.save(save_path, wav, sample_rate)
                                except Exception as e:
                                    print(f"[mmgp] torchaudio save failed: {e}, trying soundfile...", flush=True)
                                    import soundfile as sf
                                    sf.write(save_path, wav.squeeze().numpy(), sample_rate)

                                print(f"[mmgp] Audio saved to {save_path}", flush=True)
                            else:
                                raise RuntimeError("mmgp pipeline returned no audio output")
                        else:
                            # Original heartlib pipeline interface
                            pipeline_inputs = {
                                "lyrics": request.lyrics or "",  # heartlib expects string, not None
                                "tags": sound_tags,
                            }
                            if ref_audio_path:
                                pipeline_inputs["ref_audio"] = ref_audio_path
                                pipeline_inputs["muq_segment_sec"] = muq_segment_sec
                                if request.ref_audio_start_sec is not None:
                                    pipeline_inputs["ref_audio_start_sec"] = request.ref_audio_start_sec
                            else:
                                print(f"[DEBUG] No ref_audio_path found", flush=True)

                            if request.negative_tags:
                                pipeline_inputs["negative_tags"] = request.negative_tags
                                print(f"[DEBUG] Using negative_tags: {request.negative_tags}", flush=True)

                            self.pipeline.generate_with_callback(
                                pipeline_inputs,
                                max_audio_length_ms=request.duration_ms,
                                save_path=save_path,
                                topk=request.topk,
                                temperature=request.temperature,
                                cfg_scale=request.cfg_scale,
                                callback=_pipeline_callback,
                            )

                    return None

                # Notify Start
                event_manager.publish("job_update", {"job_id": job_id_str, "status": "processing"})

                # Check if torch.compile is enabled and this is first run
                is_compiling_first_run = ENABLE_TORCH_COMPILE and self._torch_compile_first_run
                compile_timer_stop = None

                if is_compiling_first_run:
                    event_manager.publish("job_progress", {
                        "job_id": job_id_str,
                        "progress": 0,
                        "msg": "Compiling model (first run only)... 0:00 elapsed"
                    })

                    # Start a background task to update elapsed time during compilation
                    compile_start_time = time.time()
                    compile_timer_stop = asyncio.Event()

                    async def update_compile_progress():
                        while not compile_timer_stop.is_set():
                            elapsed = int(time.time() - compile_start_time)
                            mins, secs = divmod(elapsed, 60)
                            event_manager.publish("job_progress", {
                                "job_id": job_id_str,
                                "progress": 0,
                                "msg": f"Compiling model (first run only)... {mins}:{secs:02d} elapsed"
                            })
                            try:
                                await asyncio.wait_for(compile_timer_stop.wait(), timeout=5.0)
                                break
                            except asyncio.TimeoutError:
                                pass

                    compile_timer_task = asyncio.create_task(update_compile_progress())
                else:
                    event_manager.publish("job_progress", {"job_id": job_id_str, "progress": 0, "msg": "Starting generation pipeline..."})

                await loop.run_in_executor(None, _run_pipeline)

                # Stop compile timer if it was running
                if compile_timer_stop:
                    compile_timer_stop.set()
                    await compile_timer_task

                # Mark first run complete
                if is_compiling_first_run:
                    self._torch_compile_first_run = False

                # 6. Update status to COMPLETED
                generation_time = time.time() - generation_start_time

                with Session(db_engine) as session:
                    job = session.exec(select(Job).where(Job.id == job_id_uuid)).one_or_none()
                    if not job:
                        logger.warning(f"Job {job_id_str} was deleted during generation. Discarding result.")
                        return

                    job.status = JobStatus.COMPLETED
                    job.audio_path = f"/audio/{output_filename}"
                    job.title = auto_title
                    job.seed = seed_to_use
                    job.generation_time_seconds = round(generation_time, 1)
                    session.add(job)
                    session.commit()
                    final_audio_path = job.audio_path
                    final_title = job.title

                logger.info(f"Job {job_id_str} completed. Saved to {save_path}")
                event_manager.publish("job_update", {"job_id": job_id_str, "status": "completed", "audio_path": final_audio_path, "title": final_title})
                event_manager.publish("job_progress", {"job_id": job_id_str, "progress": 100, "msg": "Done!"})

            except Exception as e:
                logger.error(f"Job {job_id_str} failed: {e}")
                import traceback
                traceback.print_exc()
                with Session(db_engine) as session:
                    job = session.exec(select(Job).where(Job.id == job_id_uuid)).one()
                    job.status = JobStatus.FAILED
                    job.error_msg = str(e)
                    session.add(job)
                    session.commit()
                event_manager.publish("job_update", {"job_id": job_id_str, "status": "failed", "error": str(e)})

            finally:
                # Cleanup cancellation event
                if job_id_str in self.active_jobs:
                    del self.active_jobs[job_id_str]

                # Aggressive GPU memory cleanup after each generation
                try:
                    if self.pipeline and hasattr(self.pipeline, 'mula'):
                        self.pipeline.mula.reset_caches()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    logger.info("GPU memory cleaned up after generation")
                except Exception as cleanup_err:
                    logger.warning(f"Memory cleanup warning: {cleanup_err}")

    def cancel_job(self, job_id: str):
        # Check if job is in queue (waiting)
        if job_id in self.job_queue:
            logger.info(f"Removing queued job {job_id}")
            self.job_queue.remove(job_id)
            self._broadcast_queue_update()
            return True
        # Check if job is actively processing
        if job_id in self.active_jobs:
            logger.info(f"Cancelling active job {job_id}")
            self.active_jobs[job_id].set()
            return True
        return False

    def shutdown_all(self):
        """Cancel all active jobs."""
        logger.info(f"Shutting down MusicService. Cancelling {len(self.active_jobs)} active jobs.")
        for job_id, event in list(self.active_jobs.items()):
            event.set()


class EventManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventManager, cls).__new__(cls)
            cls._instance.subscribers = []
        return cls._instance

    def subscribe(self):
        q = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self.subscribers:
            self.subscribers.remove(q)

    def publish(self, event_type: str, data: dict):
        import json
        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        for q in self.subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    def shutdown(self):
        """Broadcast shutdown signal to all subscribers to release connections."""
        msg = "event: shutdown\ndata: {}\n\n"
        for q in self.subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass


music_service = MusicService()
event_manager = EventManager()
