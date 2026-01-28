#!/usr/bin/env python3
"""
HeartMuLa Studio macOS Launcher
This script is the entry point for the PyInstaller-bundled macOS app.
It sets up the environment and launches the FastAPI server with the frontend.
"""

import os
import sys
import subprocess
import time
import shutil
import threading
from pathlib import Path
import socket

def setup_environment():
    """Set up the macOS app environment."""
    # Determine if we're running as a bundled app
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        bundle_dir = sys._MEIPASS
        app_dir = Path(bundle_dir)
    else:
        # Running as script
        app_dir = Path(__file__).parent
    
    # Set up paths for the app - ALL data goes to user's Library directory
    home_dir = Path.home()
    app_support_dir = home_dir / "Library" / "Application Support" / "HeartMuLa"
    models_dir = app_support_dir / "models"
    generated_audio_dir = app_support_dir / "generated_audio"
    ref_audio_dir = app_support_dir / "ref_audio"
    logs_dir = home_dir / "Library" / "Logs" / "HeartMuLa"
    
    # Create directories
    for directory in [app_support_dir, models_dir, generated_audio_dir, ref_audio_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # IMPORTANT: Set environment variables BEFORE importing backend modules
    # This ensures all paths point to the user's Library directory, NOT the app bundle
    os.environ["HEARTMULA_MODEL_DIR"] = str(models_dir)
    os.environ["HEARTMULA_DB_PATH"] = str(app_support_dir / "jobs.db")
    os.environ["HEARTMULA_GENERATED_AUDIO_DIR"] = str(generated_audio_dir)
    os.environ["HEARTMULA_REF_AUDIO_DIR"] = str(ref_audio_dir)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable Metal Performance Shaders
    
    # Change working directory to app bundle for frontend serving
    # Backend will use absolute paths from environment variables
    os.chdir(app_dir)
    
    print(f"Data directories configured:")
    print(f"  Models: {models_dir}")
    print(f"  Generated Audio: {generated_audio_dir}")
    print(f"  Reference Audio: {ref_audio_dir}")
    print(f"  Database: {app_support_dir / 'jobs.db'}")
    print(f"  Logs: {logs_dir}")
    
    return app_dir, logs_dir

def check_single_instance():
    """Check if another instance of the app is already running."""
    lock_file = Path.home() / "Library" / "Application Support" / "HeartMuLa" / ".lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to bind to a port to ensure single instance
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 58765))  # Random high port for lock
        return sock  # Keep socket open to maintain lock
    except OSError:
        # Port is already in use - another instance is running
        return None

def launch_server(app_dir, logs_dir):
    """Launch the FastAPI server."""
    # Import and run the FastAPI app
    sys.path.insert(0, str(app_dir))
    
    # Configure uvicorn to run the app
    import uvicorn
    from backend.app.main import app
    
    # Run the server in a thread so pywebview can take control of main thread
    def run_server():
        print(f"Starting HeartMuLa Studio server...")
        print(f"Logs directory: {logs_dir}")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True
        )
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Launch pywebview window
    try:
        import webview
        print("Opening HeartMuLa Studio window...")
        
        # Create window with custom settings
        window = webview.create_window(
            'HeartMuLa Studio',
            'http://127.0.0.1:8000',
            width=1400,
            height=900,
            resizable=True,
            fullscreen=False,
            min_size=(800, 600),
            background_color='#1a1a1a',
            text_select=True
        )
        
        # Start the webview - this blocks until window is closed
        webview.start()
        
    except ImportError:
        print("Warning: pywebview not available, falling back to browser")
        import webbrowser
        webbrowser.open("http://localhost:8000")
        # Keep the server running
        while True:
            time.sleep(1)

def main():
    """Main entry point."""
    try:
        # Check if another instance is already running
        lock_socket = check_single_instance()
        if lock_socket is None:
            print("Another instance of HeartMuLa Studio is already running.")
            print("Only one instance can be opened at a time.")
            sys.exit(0)
        
        app_dir, logs_dir = setup_environment()
        launch_server(app_dir, logs_dir)
    except KeyboardInterrupt:
        print("\nShutting down HeartMuLa Studio...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting HeartMuLa Studio: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
