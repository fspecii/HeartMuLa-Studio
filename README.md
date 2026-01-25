<p align="center">
  <img src="https://raw.githubusercontent.com/fspecii/HeartMuLa-Studio/main/frontend/public/heartmula-icon.svg" alt="HeartMuLa Studio" width="120" height="120">
</p>

<h1 align="center">HeartMuLa Studio</h1>

<p align="center">
  <strong>A professional, Suno-like music generation studio for <a href="https://github.com/HeartMuLa/heartlib">HeartLib</a></strong>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=W7-JB-Pl8So">
    <img src="https://img.shields.io/badge/▶_Watch_Demo-YouTube-FF0000?style=for-the-badge&logo=youtube" alt="Watch Demo on YouTube">
  </a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#credits">Credits</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/TypeScript-5.6-3178C6?style=flat-square&logo=typescript" alt="TypeScript">
  <img src="https://img.shields.io/badge/TailwindCSS-3.4-06B6D4?style=flat-square&logo=tailwindcss" alt="TailwindCSS">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

## Demo

<p align="center">
  <img src="preview.gif" alt="HeartMuLa Studio Preview" width="100%">
</p>

## Features

### 🎵 AI Music Generation
| Feature | Description |
|---------|-------------|
| **Full Song Generation** | Create complete songs with vocals and lyrics up to 4+ minutes |
| **Instrumental Mode** | Generate instrumental tracks without vocals |
| **Style Tags** | Define genre, mood, tempo, and instrumentation |
| **Seed Control** | Reproduce exact generations for consistency |
| **Queue System** | Queue multiple generations and process them sequentially |

### 🎨 Reference Audio (Style Transfer) `Experimental`
| Feature | Description |
|---------|-------------|
| **Audio Upload** | Use any audio file as a style reference |
| **Waveform Visualization** | Professional waveform display powered by WaveSurfer.js |
| **Region Selection** | Draggable 10-second region selector for precise style sampling |
| **Style Influence** | Adjustable slider to control reference audio influence (1-100%) |
| **Synced Playback** | Modal waveform syncs with bottom player in real-time |

> **Coming Soon: LoRA Voice Training** - We're actively developing LoRA-based voice training with exceptional results. Our early tests show voice consistency that surpasses Suno. Stay tuned for updates!

### 🎤 AI-Powered Lyrics
| Feature | Description |
|---------|-------------|
| **Lyrics Generation** | Generate lyrics from a topic using LLMs |
| **Multiple Providers** | Support for Ollama (local) and OpenRouter (cloud) |
| **Style Suggestions** | AI-suggested style tags based on your concept |
| **Prompt Enhancement** | Improve your prompts with AI assistance |

### 🎧 Professional Interface
| Feature | Description |
|---------|-------------|
| **Spotify-Inspired UI** | Clean, modern design with dark/light mode |
| **Bottom Player** | Full-featured player with waveform, volume, and progress |
| **History Feed** | Browse, search, and manage all generated tracks |
| **Likes & Playlists** | Organize favorites into custom playlists |
| **Real-time Progress** | Live generation progress with step indicators |
| **Responsive Design** | Works on desktop and mobile devices |

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, TypeScript, TailwindCSS, Framer Motion, WaveSurfer.js |
| **Backend** | FastAPI, SQLModel, SSE (Server-Sent Events) |
| **AI Engine** | [HeartLib](https://github.com/HeartMuLa/heartlib) - MuQ, MuLan, HeartCodec |
| **LLM Integration** | Ollama, OpenRouter |

## Performance Optimizations

HeartMuLa Studio includes several optimizations for faster generation and lower VRAM usage:

### 🚀 4-bit Quantization
Reduces VRAM usage from **~11GB to ~3GB** using BitsAndBytes NF4 quantization:
```bash
HEARTMULA_4BIT=true python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### ⚡ Flash Attention
Automatically configured based on your GPU:
| GPU | Flash Attention |
|-----|-----------------|
| NVIDIA SM 7.0+ (Volta, Turing, Ampere, Ada, Hopper) | ✅ Enabled |
| NVIDIA SM 6.x and older (Pascal, Maxwell) | ❌ Disabled (uses math backend) |
| AMD GPUs | ❌ Disabled (compatibility varies) |

### 🔥 torch.compile (Experimental)
Enable PyTorch 2.0+ compilation for **~2x faster inference** on supported GPUs:

```bash
# Enable torch.compile
HEARTMULA_COMPILE=true python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# With max performance (slower first run, faster subsequent runs)
HEARTMULA_COMPILE=true HEARTMULA_COMPILE_MODE=max-autotune python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

| Mode | Description |
|------|-------------|
| `default` | Good balance of compile time and performance |
| `reduce-overhead` | Faster compilation, slightly less optimal code |
| `max-autotune` | Best performance, but slowest compilation (recommended for production) |

**Requirements:**
- PyTorch 2.0+
- **Linux/WSL2**: Install Triton (`pip install triton`)
- **Windows**: Install Triton-Windows (`pip install -U 'triton-windows>=3.2,<3.3'`)

> **Note:** First generation will be slower due to compilation. Subsequent generations benefit from the compiled kernels.

### 🎯 Smart Multi-GPU Detection
Automatically selects the best GPU configuration:
- **With 4-bit quantization**: Prioritizes fastest GPU (highest compute capability)
- **Without quantization**: Prioritizes GPU with most VRAM
- HeartMuLa → Primary GPU, HeartCodec → Secondary GPU

### 📥 Auto-Download
Models are automatically downloaded from HuggingFace Hub on first run (~5GB):
- HeartMuLa (main model)
- HeartCodec (audio decoder)
- Tokenizer and generation config

## Quick Start

```bash
./start.sh
```

That's it! The system auto-detects your GPU and downloads models on first run.

Open http://localhost:5173

## Docker (Recommended)

The easiest way to run HeartMuLa Studio - no Python/Node setup required.

### Prerequisites
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA GPU with 10GB+ VRAM

### Quick Start with Docker

```bash
# Clone and start (uses pre-built image from GitHub Container Registry)
git clone https://github.com/fspecii/HeartMuLa-Studio.git
cd HeartMuLa-Studio
docker compose up -d

# View logs (watch model download progress on first run)
docker compose logs -f
```

Open **http://localhost:8000**

### Alternative: Pull and Run Directly

```bash
# Create directories for persistent data
mkdir -p backend/models backend/generated_audio backend/ref_audio

# Run the pre-built image (Docker Hub)
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v ./backend/models:/app/backend/models \
  -v ./backend/generated_audio:/app/backend/generated_audio \
  -v ./backend/ref_audio:/app/backend/ref_audio \
  --name heartmula-studio \
  ambsd/heartmula-studio:latest
```

**Available registries:**
- Docker Hub: `ambsd/heartmula-studio:latest`
- GitHub: `ghcr.io/fspecii/heartmula-studio:latest`

### What Happens on First Run

1. Docker builds the image (~10GB, includes CUDA + PyTorch)
2. Models are automatically downloaded from HuggingFace (~5GB)
3. Container starts with GPU auto-detection
4. Frontend + API served on port 8000

### Persistent Data

All your data is preserved across container restarts:

| Data | Location | Description |
|------|----------|-------------|
| **Generated Music** | `./backend/generated_audio/` | Your MP3 files (accessible from host) |
| **Models** | `./backend/models/` | Downloaded AI models (~5GB) |
| **Reference Audio** | `./backend/ref_audio/` | Uploaded style references |
| **Song History** | Docker volume `heartmula-db` | Database with all your generations |

### Docker Commands

```bash
# Start
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f

# Rebuild after updates
docker compose build --no-cache
docker compose up -d

# Reset database (fresh start)
docker compose down -v
docker compose up -d
```

### Docker Configuration

Override settings in `docker-compose.yml`:

```yaml
environment:
  - HEARTMULA_4BIT=true                  # Force 4-bit quantization
  - HEARTMULA_SEQUENTIAL_OFFLOAD=true    # Force model swapping (low VRAM)

volumes:
  # Use existing models from another location (e.g., ComfyUI)
  - /path/to/comfyui/models/heartmula:/app/backend/models
```

### Using Ollama with Docker

To use Ollama (running on host) for AI lyrics generation:

1. **Ollama is auto-configured** - The container uses `host.docker.internal` to reach Ollama on your host machine
2. **Just run Ollama normally** on your host (not in Docker)
3. The container will automatically connect to `http://host.docker.internal:11434`

**Custom Ollama URL:**
```yaml
environment:
  - OLLAMA_HOST=http://your-ollama-server:11434
```

## Prerequisites

- **Python** 3.10 or higher
- **Node.js** 18 or higher
- **CUDA GPU** with 10GB+ VRAM
- **Git** for cloning the repository

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/fspecii/HeartMuLa-Studio.git
cd HeartMuLa-Studio
```

### 2. Backend Setup

```bash
# Create virtual environment in root folder
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

> **Note:** HeartLib models (~5GB) will be downloaded automatically from HuggingFace on first run.

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Build for production
npm run build
```

## Usage

### Start the Backend

```bash
source venv/bin/activate  # Windows: venv\Scripts\activate

# Single GPU
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Multi-GPU (recommended for 2+ GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### Start the Frontend

**Development mode:**
```bash
cd frontend
npm run dev
```

**Production mode:**
```bash
# Serve the dist folder with any static server
npx serve dist -l 5173
```

### Access the Application

| Mode | URL |
|------|-----|
| Development | http://localhost:5173 |
| Production | http://localhost:8000 |

## Configuration

### Environment Variables

Create a `.env` file in the `backend` directory:

```env
# OpenRouter API (for cloud LLM)
OPENROUTER_API_KEY=your_api_key_here

# Ollama (for local LLM)
OLLAMA_HOST=http://localhost:11434
```

**HeartMuLa Configuration (set when running):**

| Variable | Default | Description |
|----------|---------|-------------|
| `HEARTMULA_MODEL_DIR` | `backend/models` | Custom model directory (share with ComfyUI, etc.) |
| `HEARTMULA_4BIT` | `auto` | 4-bit quantization: `auto`, `true`, or `false` |
| `HEARTMULA_SEQUENTIAL_OFFLOAD` | `auto` | Model swapping for low VRAM: `auto`, `true`, or `false` |
| `HEARTMULA_COMPILE` | `false` | torch.compile for ~2x faster inference: `true` or `false` |
| `HEARTMULA_COMPILE_MODE` | `default` | Compile mode: `default`, `reduce-overhead`, or `max-autotune` |
| `HEARTMULA_VERSION` | `RL-3B-20260123` | Model version (latest RL-tuned model) |
| `CUDA_VISIBLE_DEVICES` | all GPUs | Specify which GPUs to use (e.g., `0,1`) |

**Example: Use existing models from ComfyUI:**
```bash
HEARTMULA_MODEL_DIR=/path/to/comfyui/models/heartmula ./start.sh
```

### GPU Auto-Configuration

HeartMuLa Studio **automatically detects** your GPU VRAM and selects the optimal configuration:

| Your VRAM | Auto-Selected Mode | Speed | Example GPUs |
|-----------|-------------------|-------|--------------|
| **20GB+** | Full Precision | ~7 fps | RTX 4090, RTX 3090 Ti, A6000 |
| **14-20GB** | 4-bit Quantized | ~7 fps | RTX 4060 Ti 16GB, RTX 3090 |
| **10-14GB** | 4-bit + Model Swap | ~4 fps (+70s/song) | RTX 3060 12GB, RTX 4060 8GB |
| **<10GB** | Not supported | - | Insufficient VRAM |

**Multi-GPU:** Automatically detected and used. HeartMuLa goes to fastest GPU (Flash Attention), HeartCodec to largest VRAM GPU.

### Start Options

```bash
./start.sh                # Auto-detect (recommended)
./start.sh --force-4bit   # Force 4-bit quantization
./start.sh --force-swap   # Force model swapping (low VRAM mode)
./start.sh --help         # Show all options
```

### Manual Configuration (Advanced)

Override auto-detection with environment variables:

```bash
# Force specific settings
HEARTMULA_4BIT=true HEARTMULA_SEQUENTIAL_OFFLOAD=false ./start.sh

# Or run directly
HEARTMULA_4BIT=true python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

| Variable | Values | Description |
|----------|--------|-------------|
| `HEARTMULA_4BIT` | `auto`, `true`, `false` | 4-bit quantization (default: auto) |
| `HEARTMULA_SEQUENTIAL_OFFLOAD` | `auto`, `true`, `false` | Model swapping for low VRAM (default: auto) |
| `CUDA_VISIBLE_DEVICES` | `0`, `0,1`, etc. | Select specific GPUs |

**Memory Optimization:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### LLM Setup (Optional)

For AI-powered lyrics generation:

**Option A: Ollama (Local)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2
```

**Option B: OpenRouter (Cloud)**
1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Add it to your `.env` file

## Project Structure

```
HeartMuLa-Studio/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application & routes
│   │   ├── models.py            # Pydantic/SQLModel schemas
│   │   └── services/
│   │       ├── music_service.py # HeartLib integration
│   │       └── llm_service.py   # LLM providers
│   ├── generated_audio/         # Output MP3 files
│   ├── ref_audio/               # Uploaded reference audio
│   ├── jobs.db                  # SQLite database
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ComposerSidebar.tsx    # Main generation form
│   │   │   ├── BottomPlayer.tsx       # Audio player
│   │   │   ├── RefAudioRegionModal.tsx # Waveform selector
│   │   │   ├── HistoryFeed.tsx        # Track history
│   │   │   └── ...
│   │   ├── App.tsx              # Main application
│   │   └── api.ts               # Backend API client
│   ├── public/
│   └── package.json
├── preview.gif
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate/music` | Start music generation |
| `POST` | `/generate/lyrics` | Generate lyrics with LLM |
| `POST` | `/upload/ref_audio` | Upload reference audio |
| `GET` | `/history` | Get generation history |
| `GET` | `/jobs/{id}` | Get job status |
| `GET` | `/events` | SSE stream for real-time updates |
| `GET` | `/audio/{path}` | Stream generated audio |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | System should auto-detect. Try `./start.sh --force-swap` or reduce duration |
| Models not downloading | Check internet connection and disk space (~5GB needed in `backend/models/`) |
| Frontend can't connect | Ensure backend is running on port 8000 |
| LLM not working | Check Ollama is running or OpenRouter API key is set in `backend/.env` |
| Only one GPU detected | Set `CUDA_VISIBLE_DEVICES=0,1` explicitly when starting backend |
| Slow generation | Check logs: `tail -f /tmp/heartmula_backend.log` for GPU config |

### Models Location

Models are auto-downloaded to `backend/models/` (~5GB total):
```
backend/models/
├── HeartMuLa-oss-RL-3B-20260123/   # Main model
├── HeartCodec-oss/                  # Audio codec
├── tokenizer.json
└── gen_config.json
```

## Credits

- **[HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib)** - The open-source AI music generation engine
- **[mainza-ai/milimomusic](https://github.com/mainza-ai/milimomusic)** - Inspiration for the backend architecture
- **[WaveSurfer.js](https://wavesurfer.xyz/)** - Audio waveform visualization

## License

This project is open source under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  Made with ❤️ for the open-source AI music community
</p>
