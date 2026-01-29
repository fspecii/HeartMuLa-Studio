# Apple Metal (MPS) GPU Optimization

This document describes the optimizations made to enable fast music generation on Apple Silicon (M1/M2/M3) devices using Metal Performance Shaders (MPS).

## Problem

PR #11 fixed a blocker but generation was running very slowly on Apple Silicon, most likely falling back to CPU instead of utilizing the GPU.

## Root Cause

The code was using `torch.float32` precision for models on MPS devices. While MPS supports float32, it is **significantly slower** than float16 operations. MPS is optimized for float16 (half-precision) operations which leverage the GPU's native capabilities.

## Solution

### 1. Float16 Precision (Critical Performance Fix)

Changed model dtype from `torch.float32` to `torch.float16` for both HeartMuLa and HeartCodec models when running on MPS devices.

**Why this matters:**
- MPS has native hardware acceleration for float16 operations
- float32 operations on MPS may fall back to slower execution paths
- float16 on MPS is typically **2-4x faster** than float32
- Memory usage is also reduced by half

### 2. Explicit Device Management

Added verification and automatic correction for model device placement:
- Verify models are loaded on MPS after initialization
- Automatically move models to MPS if they end up on wrong device
- Explicitly set pipeline device and dtype attributes

### 3. MPS Fallback Configuration

Set `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable to enable graceful CPU fallback for any operations not yet supported by MPS, preventing crashes while maintaining GPU acceleration for supported operations.

### 4. Consistent Dtype Handling

Ensured that lazy-loaded models (like HeartCodec) use the same dtype as the pipeline configuration instead of hardcoded values.

## Technical Details

### Changes Made

1. **`backend/app/services/music_service.py`** (top of file):
   - Added MPS configuration at module import time
   - Set `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable

2. **Model Loading** (MPS pipeline initialization):
   - Changed from `torch.float32` to `torch.float16` for MPS
   - Added device verification after model loading
   - Explicitly set pipeline attributes: `mula_device`, `codec_device`, `mula_dtype`, `codec_dtype`
   - Added automatic device correction if models are on wrong device

3. **Lazy Codec Loading** (codec loading function):
   - Use `pipeline.codec_dtype` instead of hardcoded `torch.float32`
   - Added MPS-specific logging

4. **Generation Logging** (generation start):
   - Added diagnostic logging to show device and dtype at generation start

### Performance Impact

Expected performance improvements on Apple Silicon:
- **2-4x faster generation** compared to float32
- Reduced memory usage (float16 uses half the memory of float32)
- Full GPU utilization instead of CPU fallback

## Testing

To verify the optimizations are working:

1. Check the logs during model loading - you should see:
   ```
   [Apple Metal] Loading models with float16 precision for optimal MPS performance
   [Apple Metal] HeartMuLa model device: mps:0
   [Apple Metal] HeartCodec model device: mps:0
   [Apple Metal] MPS pipeline loaded successfully with float16 precision
   ```

2. During generation, you should see:
   ```
   [Generation] Starting generation on device: mps:0 (dtype: torch.float16)
   ```

3. Monitor Activity Monitor → GPU History - you should see GPU utilization during generation

## MPS Compatibility Notes

- **Supported Operations**: Most PyTorch operations work well on MPS
- **Float16 vs Float32**: MPS strongly prefers float16 for performance
- **Bfloat16**: Not supported on MPS, use float16 instead
- **Quantization**: 4-bit quantization (BitsAndBytes) is CUDA-only, not available on MPS
- **Torch.compile**: Not yet optimized for MPS, disabled for Apple Silicon
- **Unified Memory**: MPS uses unified memory architecture, no explicit VRAM limits

## Future Optimizations

Potential areas for further optimization:
1. Profile specific operations to identify any remaining CPU fallbacks
2. Consider using Metal Performance Shaders directly for certain operations
3. Explore torch.compile support as it matures for MPS
4. Investigate mixed precision training/inference techniques

## References

- [PyTorch MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
- [PyTorch Float16 on MPS](https://github.com/pytorch/pytorch/issues/77764)
