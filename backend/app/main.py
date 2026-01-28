import asyncio
import os
import uuid as uuid_module
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env file
load_dotenv("backend/.env")
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, Session, create_engine, select
from typing import List
from datetime import datetime, timezone
from uuid import UUID

from backend.app.models import (
    Job, JobStatus, GenerationRequest, LyricsRequest, EnhancePromptRequest, InspirationRequest,
    LikedSong, Playlist, PlaylistSong, CreatePlaylistRequest, UpdatePlaylistRequest, AddToPlaylistRequest,
    GPUSettingsRequest, GPUSettingsResponse, GPUStatusResponse, StartupStatusResponse, ModelReloadResponse,
    LLMSettingsRequest, LLMSettingsResponse
)
from backend.app.services.music_service import music_service
from backend.app.services.llm_service import LLMService

# Database - configurable path for Docker support
sqlite_file_name = os.environ.get("HEARTMULA_DB_PATH", "backend/jobs.db")
# Ensure directory exists for Docker volume mount
os.makedirs(os.path.dirname(sqlite_file_name) if os.path.dirname(sqlite_file_name) else ".", exist_ok=True)
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def run_migrations():
    """Run simple database migrations for new columns."""
    from sqlalchemy import text
    with engine.connect() as conn:
        # Add generation_time_seconds column if it doesn't exist
        try:
            conn.execute(text("ALTER TABLE job ADD COLUMN generation_time_seconds REAL"))
            conn.commit()
            print("[Migration] Added generation_time_seconds column to job table")
        except Exception:
            pass  # Column already exists

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    run_migrations()
    # Start model initialization in background - server starts immediately
    # Frontend can connect and show progress via SSE
    asyncio.create_task(music_service.initialize_with_progress())
    yield
    # Shutdown Event Manager (Closes SSE connections)
    event_manager.shutdown()
    music_service.shutdown_all()

app = FastAPI(lifespan=lifespan, title="HeartMuLa Music API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files (Audio Serving)
# Use environment variable for generated audio directory (for macOS app support)
GENERATED_AUDIO_DIR = os.environ.get("HEARTMULA_GENERATED_AUDIO_DIR", "backend/generated_audio")
os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=GENERATED_AUDIO_DIR), name="audio")

# Reference Audio Storage
# Use environment variable for reference audio directory (for macOS app support)
REF_AUDIO_DIR = os.environ.get("HEARTMULA_REF_AUDIO_DIR", "backend/ref_audio")
os.makedirs(REF_AUDIO_DIR, exist_ok=True)
app.mount("/ref_audio", StaticFiles(directory=REF_AUDIO_DIR), name="ref_audio")

# --- Routes ---

@app.post("/upload/ref_audio")
async def upload_ref_audio(file: UploadFile = File(...)):
    """Upload a reference audio file for style conditioning."""
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/flac", "audio/ogg"]
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Allowed: mp3, wav, flac, ogg")

    # Generate unique ID for the file
    file_id = str(uuid_module.uuid4())

    # Get file extension
    original_name = file.filename or "audio.mp3"
    ext = os.path.splitext(original_name)[1] or ".mp3"

    # Save file
    file_path = os.path.join(REF_AUDIO_DIR, f"{file_id}{ext}")
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    return {
        "id": file_id,
        "filename": original_name,
        "path": f"/ref_audio/{file_id}{ext}",
        "size": len(contents)
    }


@app.delete("/upload/ref_audio/{file_id}")
async def delete_ref_audio(file_id: str):
    """Delete a previously uploaded reference audio file."""
    # Find and delete the file (could have various extensions)
    for ext in [".mp3", ".wav", ".flac", ".ogg"]:
        file_path = os.path.join(REF_AUDIO_DIR, f"{file_id}{ext}")
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"status": "deleted", "id": file_id}

    raise HTTPException(status_code=404, detail="Reference audio not found")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": music_service.pipeline is not None}

@app.get("/models/lyrics")
def get_lyrics_models():
    return {"models": LLMService.get_models()}

@app.get("/languages")
def get_languages():
    return {"languages": LLMService.get_supported_languages()}

@app.post("/generate/enhance_prompt")
def enhance_prompt(req: EnhancePromptRequest):
    try:
        result = LLMService.enhance_prompt(req.concept, req.model_name, req.provider)
        return result
    except Exception as e:
        # Fallback
        return {"topic": req.concept, "tags": "Pop"}

@app.post("/generate/evaluate_inspiration")
def generate_inspiration(req: InspirationRequest):
    try:
        result = LLMService.generate_inspiration(req.model_name, req.provider)
        return result
    except Exception as e:
        return {"topic": "A futuristic city in the clouds", "tags": "Electronic, ambient, sci-fi"}

@app.post("/generate/styles")
def generate_styles(req: InspirationRequest):
    # Reusing InspirationRequest since we just need the model_name
    try:
        styles = LLMService.generate_styles_list(req.model_name)
        return {"styles": styles}
    except Exception:
        return {"styles": ["Pop", "Rock", "Jazz"]} # Fallback

@app.post("/generate/lyrics")
def generate_lyrics(req: LyricsRequest):
    try:
        result = LLMService.generate_lyrics(req.topic, req.model_name, req.seed_lyrics, req.provider, req.language)
        return {
            "lyrics": result["lyrics"],
            "suggested_topic": result.get("suggested_topic", req.topic),
            "suggested_tags": result.get("suggested_tags", "Pop, Melodic")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/music")
async def generate_music(req: GenerationRequest, background_tasks: BackgroundTasks):
    # Create Job Record
    seed_val = req.seed
    if seed_val is None:
         import random
         seed_val = random.randint(0, 2**32 - 1)
         
    job = Job(prompt=req.prompt, lyrics=req.lyrics, duration_ms=req.duration_ms, tags=req.tags, seed=seed_val)
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    
    # Enqueue Background Task
    background_tasks.add_task(music_service.generate_task, job.id, req, engine)
    
    return {"job_id": job.id, "status": job.status}

@app.get("/jobs/{job_id}", response_model=Job)
def get_job_status(job_id: UUID):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

@app.get("/history", response_model=List[Job])
def get_history():
    with Session(engine) as session:
        jobs = session.exec(select(Job).order_by(Job.created_at.desc())).all()
        return jobs

@app.patch("/jobs/{job_id}", response_model=Job)
def rename_job(job_id: UUID, upgrade: dict):
    # Minimal schema for update, expecting {"title": "new name"}
    new_title = upgrade.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Title is required")
        
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.title = new_title
        session.add(job)
        session.commit()
        session.refresh(job)
        return job

@app.get("/download_track/{job_id}")
def download_track(job_id: UUID):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job or not job.audio_path:
            raise HTTPException(status_code=404, detail="Track not found")
            
        # audio_path is "/audio/filename.mp3" -> use GENERATED_AUDIO_DIR
        filename = job.audio_path.replace("/audio/", "")
        file_path = os.path.join(GENERATED_AUDIO_DIR, filename)
        
        # Sanitize Title for Filename
        import re
        safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', job.title or "untitled")
        safe_title = safe_title.strip().replace(" ", "_")
        download_name = f"{safe_title}.mp3"
        
        return FileResponse(file_path, media_type="audio/mpeg", filename=download_name)

@app.delete("/jobs/{job_id}")
def delete_job(job_id: UUID):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Delete audio file if exists
        if job.audio_path:
            # audio_path is like "/audio/filename.mp3"
            # Use GENERATED_AUDIO_DIR to find the file
            filename = job.audio_path.replace("/audio/", "")
            file_path = os.path.join(GENERATED_AUDIO_DIR, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
        session.delete(job)
        session.commit()
        return {"status": "deleted", "id": job_id}

@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: UUID):
    # Try to cancel running task via service
    if music_service.cancel_job(str(job_id)):
        return {"status": "cancelling", "id": job_id}
    
    # If not running, maybe update status in DB directly?
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job and job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            job.status = JobStatus.FAILED
            job.error_msg = "Cancelled by user"
            session.add(job)
            session.commit()
            return {"status": "cancelled", "id": job_id}
            
    raise HTTPException(status_code=400, detail="Job not active or already completed")

from fastapi.responses import StreamingResponse
from backend.app.services.music_service import event_manager

@app.get("/events")
async def events():
    async def event_generator():
        q = event_manager.subscribe()
        try:
            while True:
                # Wait for new event using asyncio.wait_for to allow checking client disconnected
                # actually Queue.get is async so it yields control
                try:
                    data = await asyncio.wait_for(q.get(), timeout=1.0)
                    if "event: shutdown" in data:
                        break
                    yield data
                except asyncio.TimeoutError:
                    # Wake up loop to check for cancellation or keep-alive
                    # yield ": keep-alive\n\n" # Optional: send comment to keep client connection alive
                    continue
        except asyncio.CancelledError:
             # Server shutting down
             pass
        except Exception:
            pass
        finally:
            event_manager.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============== LIKES (Favorites) ==============

@app.post("/songs/{job_id}/like")
def like_song(job_id: UUID):
    with Session(engine) as session:
        # Check if job exists
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Song not found")

        # Check if already liked
        existing = session.exec(select(LikedSong).where(LikedSong.job_id == job_id)).first()
        if existing:
            return {"status": "already_liked", "job_id": str(job_id)}

        # Add like
        liked = LikedSong(job_id=job_id)
        session.add(liked)
        session.commit()
        return {"status": "liked", "job_id": str(job_id)}


@app.delete("/songs/{job_id}/like")
def unlike_song(job_id: UUID):
    with Session(engine) as session:
        liked = session.exec(select(LikedSong).where(LikedSong.job_id == job_id)).first()
        if not liked:
            raise HTTPException(status_code=404, detail="Song not in favorites")

        session.delete(liked)
        session.commit()
        return {"status": "unliked", "job_id": str(job_id)}


@app.get("/songs/liked")
def get_liked_songs():
    with Session(engine) as session:
        # Get all liked song IDs
        liked_entries = session.exec(select(LikedSong).order_by(LikedSong.liked_at.desc())).all()
        liked_job_ids = [entry.job_id for entry in liked_entries]

        # Get the actual job details
        if not liked_job_ids:
            return {"songs": [], "liked_ids": []}

        jobs = session.exec(select(Job).where(Job.id.in_(liked_job_ids))).all()
        # Sort by liked order
        job_map = {job.id: job for job in jobs}
        sorted_jobs = [job_map[jid] for jid in liked_job_ids if jid in job_map]

        return {"songs": sorted_jobs, "liked_ids": [str(jid) for jid in liked_job_ids]}


@app.get("/songs/liked/ids")
def get_liked_song_ids():
    """Quick endpoint to get just the IDs of liked songs"""
    with Session(engine) as session:
        liked_entries = session.exec(select(LikedSong)).all()
        return {"liked_ids": [str(entry.job_id) for entry in liked_entries]}


# ============== PLAYLISTS ==============

@app.get("/playlists")
def get_playlists():
    with Session(engine) as session:
        playlists = session.exec(select(Playlist).order_by(Playlist.updated_at.desc())).all()

        # Get song count for each playlist
        result = []
        for playlist in playlists:
            song_count = len(session.exec(select(PlaylistSong).where(PlaylistSong.playlist_id == playlist.id)).all())
            result.append({
                "id": str(playlist.id),
                "name": playlist.name,
                "description": playlist.description,
                "cover_seed": playlist.cover_seed,
                "song_count": song_count,
                "created_at": playlist.created_at.isoformat(),
                "updated_at": playlist.updated_at.isoformat()
            })

        return {"playlists": result}


@app.post("/playlists")
def create_playlist(req: CreatePlaylistRequest):
    with Session(engine) as session:
        import uuid
        playlist = Playlist(
            name=req.name,
            description=req.description,
            cover_seed=str(uuid.uuid4())  # Random seed for procedural cover
        )
        session.add(playlist)
        session.commit()
        session.refresh(playlist)

        return {
            "id": str(playlist.id),
            "name": playlist.name,
            "description": playlist.description,
            "cover_seed": playlist.cover_seed,
            "created_at": playlist.created_at.isoformat()
        }


@app.get("/playlists/{playlist_id}")
def get_playlist(playlist_id: UUID):
    with Session(engine) as session:
        playlist = session.get(Playlist, playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")

        # Get songs in playlist with order
        playlist_songs = session.exec(
            select(PlaylistSong)
            .where(PlaylistSong.playlist_id == playlist_id)
            .order_by(PlaylistSong.position)
        ).all()

        # Get job details for each song
        job_ids = [ps.job_id for ps in playlist_songs]
        jobs = session.exec(select(Job).where(Job.id.in_(job_ids))).all() if job_ids else []
        job_map = {job.id: job for job in jobs}

        songs = []
        for ps in playlist_songs:
            if ps.job_id in job_map:
                job = job_map[ps.job_id]
                songs.append({
                    "job": job,
                    "position": ps.position,
                    "added_at": ps.added_at.isoformat()
                })

        return {
            "id": str(playlist.id),
            "name": playlist.name,
            "description": playlist.description,
            "cover_seed": playlist.cover_seed,
            "songs": songs,
            "song_count": len(songs),
            "created_at": playlist.created_at.isoformat(),
            "updated_at": playlist.updated_at.isoformat()
        }


@app.patch("/playlists/{playlist_id}")
def update_playlist(playlist_id: UUID, req: UpdatePlaylistRequest):
    with Session(engine) as session:
        playlist = session.get(Playlist, playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")

        if req.name is not None:
            playlist.name = req.name
        if req.description is not None:
            playlist.description = req.description
        playlist.updated_at = datetime.now(timezone.utc)

        session.add(playlist)
        session.commit()
        session.refresh(playlist)

        return {
            "id": str(playlist.id),
            "name": playlist.name,
            "description": playlist.description,
            "updated_at": playlist.updated_at.isoformat()
        }


@app.delete("/playlists/{playlist_id}")
def delete_playlist(playlist_id: UUID):
    with Session(engine) as session:
        playlist = session.get(Playlist, playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")

        # Delete all playlist songs
        playlist_songs = session.exec(select(PlaylistSong).where(PlaylistSong.playlist_id == playlist_id)).all()
        for ps in playlist_songs:
            session.delete(ps)

        session.delete(playlist)
        session.commit()

        return {"status": "deleted", "id": str(playlist_id)}


@app.post("/playlists/{playlist_id}/songs")
def add_song_to_playlist(playlist_id: UUID, req: AddToPlaylistRequest):
    with Session(engine) as session:
        # Verify playlist exists
        playlist = session.get(Playlist, playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")

        job_id = UUID(req.job_id)

        # Verify song exists
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Song not found")

        # Check if already in playlist
        existing = session.exec(
            select(PlaylistSong)
            .where(PlaylistSong.playlist_id == playlist_id)
            .where(PlaylistSong.job_id == job_id)
        ).first()

        if existing:
            return {"status": "already_in_playlist", "playlist_id": str(playlist_id), "job_id": req.job_id}

        # Get next position
        max_pos = session.exec(
            select(PlaylistSong.position)
            .where(PlaylistSong.playlist_id == playlist_id)
            .order_by(PlaylistSong.position.desc())
        ).first()
        next_pos = (max_pos or 0) + 1

        # Add to playlist
        playlist_song = PlaylistSong(playlist_id=playlist_id, job_id=job_id, position=next_pos)
        session.add(playlist_song)

        # Update playlist timestamp
        playlist.updated_at = datetime.now(timezone.utc)
        session.add(playlist)

        session.commit()

        return {"status": "added", "playlist_id": str(playlist_id), "job_id": req.job_id, "position": next_pos}


@app.delete("/playlists/{playlist_id}/songs/{job_id}")
def remove_song_from_playlist(playlist_id: UUID, job_id: UUID):
    with Session(engine) as session:
        playlist_song = session.exec(
            select(PlaylistSong)
            .where(PlaylistSong.playlist_id == playlist_id)
            .where(PlaylistSong.job_id == job_id)
        ).first()

        if not playlist_song:
            raise HTTPException(status_code=404, detail="Song not in playlist")

        session.delete(playlist_song)

        # Update playlist timestamp
        playlist = session.get(Playlist, playlist_id)
        if playlist:
            playlist.updated_at = datetime.now(timezone.utc)
            session.add(playlist)

        session.commit()

        return {"status": "removed", "playlist_id": str(playlist_id), "job_id": str(job_id)}


# ============== SETTINGS & STARTUP STATUS ==============

@app.get("/settings/startup/status", response_model=StartupStatusResponse)
def get_startup_status():
    """Get current startup/initialization status."""
    return music_service.get_startup_status()


@app.get("/settings/gpu/status", response_model=GPUStatusResponse)
def get_gpu_status():
    """Get GPU hardware information."""
    return music_service.get_gpu_info()


@app.get("/settings/gpu", response_model=GPUSettingsResponse)
def get_gpu_settings():
    """Get current GPU settings."""
    return music_service.current_settings


@app.put("/settings/gpu", response_model=GPUSettingsResponse)
def update_gpu_settings(settings: GPUSettingsRequest):
    """Update GPU settings (does not reload models)."""
    if settings.quantization_4bit is not None:
        music_service.current_settings["quantization_4bit"] = settings.quantization_4bit
    if settings.sequential_offload is not None:
        music_service.current_settings["sequential_offload"] = settings.sequential_offload
    if settings.torch_compile is not None:
        music_service.current_settings["torch_compile"] = settings.torch_compile
    if settings.torch_compile_mode is not None:
        music_service.current_settings["torch_compile_mode"] = settings.torch_compile_mode
    # Persist settings to disk
    music_service._save_settings()
    return music_service.current_settings


@app.post("/settings/gpu/reload", response_model=ModelReloadResponse)
async def reload_models(settings: GPUSettingsRequest, background_tasks: BackgroundTasks):
    """Reload models with new settings."""
    # Check if models are currently loading
    if music_service.is_loading:
        raise HTTPException(status_code=409, detail="Models are currently loading")

    # Check if a job is processing
    if len(music_service.active_jobs) > 0:
        raise HTTPException(status_code=409, detail="Cannot reload while a job is processing")

    if len(music_service.job_queue) > 0:
        raise HTTPException(status_code=409, detail="Cannot reload while jobs are queued")

    # Convert settings to dict
    new_settings = {}
    if settings.quantization_4bit is not None:
        new_settings["quantization_4bit"] = settings.quantization_4bit
    if settings.sequential_offload is not None:
        new_settings["sequential_offload"] = settings.sequential_offload
    if settings.torch_compile is not None:
        new_settings["torch_compile"] = settings.torch_compile
    if settings.torch_compile_mode is not None:
        new_settings["torch_compile_mode"] = settings.torch_compile_mode

    # Start reload in background
    background_tasks.add_task(music_service.reload_models, new_settings)

    return {"status": "reloading", "message": "Model reload started"}


# ============== LLM SETTINGS ==============

@app.get("/settings/llm", response_model=LLMSettingsResponse)
def get_llm_settings():
    """Get current LLM provider settings."""
    settings = LLMService.get_settings()
    # Mask API keys for security (show only last 4 chars)
    openrouter_key = settings.get("openrouter_api_key", "")
    masked_openrouter_key = f"***{openrouter_key[-4:]}" if openrouter_key and len(openrouter_key) > 4 else ""

    custom_api_key = settings.get("custom_api_key", "")
    masked_custom_key = f"***{custom_api_key[-4:]}" if custom_api_key and len(custom_api_key) > 4 else ""

    return {
        "ollama_host": settings.get("ollama_host", ""),
        "openrouter_api_key": masked_openrouter_key,
        "ollama_available": LLMService.check_ollama_available(),
        "openrouter_available": LLMService.check_openrouter_available(),
        "custom_api_base_url": settings.get("custom_api_base_url", ""),
        "custom_api_key": masked_custom_key,
        "custom_api_model": settings.get("custom_api_model", ""),
        "custom_api_available": LLMService.check_custom_api_available()
    }


@app.put("/settings/llm", response_model=LLMSettingsResponse)
def update_llm_settings(settings: LLMSettingsRequest):
    """Update LLM provider settings."""
    # Update LLMService settings
    if settings.ollama_host is not None:
        LLMService.update_settings(ollama_host=settings.ollama_host)
        music_service.current_settings["ollama_host"] = settings.ollama_host

    if settings.openrouter_api_key is not None:
        LLMService.update_settings(openrouter_api_key=settings.openrouter_api_key)
        music_service.current_settings["openrouter_api_key"] = settings.openrouter_api_key

    if settings.custom_api_base_url is not None:
        LLMService.update_settings(custom_api_base_url=settings.custom_api_base_url)
        music_service.current_settings["custom_api_base_url"] = settings.custom_api_base_url

    if settings.custom_api_key is not None:
        LLMService.update_settings(custom_api_key=settings.custom_api_key)
        music_service.current_settings["custom_api_key"] = settings.custom_api_key

    if settings.custom_api_model is not None:
        LLMService.update_settings(custom_api_model=settings.custom_api_model)
        music_service.current_settings["custom_api_model"] = settings.custom_api_model

    # Save to persistent storage
    music_service._save_settings()

    # Return updated settings
    current = LLMService.get_settings()
    openrouter_key = current.get("openrouter_api_key", "")
    masked_openrouter_key = f"***{openrouter_key[-4:]}" if openrouter_key and len(openrouter_key) > 4 else ""

    custom_api_key = current.get("custom_api_key", "")
    masked_custom_key = f"***{custom_api_key[-4:]}" if custom_api_key and len(custom_api_key) > 4 else ""

    return {
        "ollama_host": current.get("ollama_host", ""),
        "openrouter_api_key": masked_openrouter_key,
        "ollama_available": LLMService.check_ollama_available(),
        "openrouter_available": LLMService.check_openrouter_available(),
        "custom_api_base_url": current.get("custom_api_base_url", ""),
        "custom_api_key": masked_custom_key,
        "custom_api_model": current.get("custom_api_model", ""),
        "custom_api_available": LLMService.check_custom_api_available()
    }


# ============== FRONTEND STATIC FILES (Docker Production) ==============
# Serve frontend static files if the dist folder exists (Docker deployment)
FRONTEND_DIST = "frontend/dist"
if os.path.exists(FRONTEND_DIST):
    # Serve static assets (js, css, images)
    app.mount("/assets", StaticFiles(directory=f"{FRONTEND_DIST}/assets"), name="frontend_assets")

    # Catch-all route for SPA - must be last
    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        # Check if it's a static file
        file_path = os.path.join(FRONTEND_DIST, path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA routing
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_graceful_shutdown=1)
