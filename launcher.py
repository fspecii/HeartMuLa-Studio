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
import urllib.request
import urllib.error
import atexit

# Single instance lock port
SINGLE_INSTANCE_PORT = 58765

# Global references for cleanup
_server_thread = None
_lock_socket = None
_cleanup_done = False

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
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable MPS memory limit to prevent OOM errors
    
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
    # Try to bind to a port to ensure single instance
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', SINGLE_INSTANCE_PORT))
        return sock  # Keep socket open to maintain lock
    except OSError:
        # Port is already in use - another instance is running
        return None

def cleanup():
    """Cleanup resources on shutdown. Idempotent - safe to call multiple times."""
    global _lock_socket, _cleanup_done
    
    # Prevent multiple cleanup calls
    if _cleanup_done:
        return
    _cleanup_done = True
    
    print("\nCleaning up resources...")
    
    # Close the lock socket
    if _lock_socket:
        try:
            _lock_socket.close()
            print("✓ Released instance lock")
        except (OSError, Exception):
            pass
    
    # Note: Server thread is daemon and will be terminated automatically
    print("✓ Server shutdown initiated")
    print("Goodbye!")

def wait_for_server(url='http://127.0.0.1:8000/health', timeout=30):
    """Wait for the server to be ready by polling the health endpoint."""
    print("Waiting for server to start...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = urllib.request.urlopen(url, timeout=1)
            if response.getcode() == 200:
                print("Server is ready!")
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            # Server not ready yet, wait a bit
            time.sleep(0.5)
    print(f"Warning: Server did not respond within {timeout} seconds")
    return False

def launch_server(app_dir, logs_dir):
    """Launch the FastAPI server."""
    global _server_thread
    
    # Import and run the FastAPI app
    sys.path.insert(0, str(app_dir))
    
    # Configure uvicorn to run the app
    import uvicorn
    from backend.app.main import app
    
    # Set up log file to capture all output
    log_file_path = logs_dir / "console.log"
    
    # Custom print wrapper that writes to both stdout and log file
    import builtins
    _original_print = builtins.print
    
    def _logged_print(*args, **kwargs):
        """Print that also writes to log file."""
        # Call original print
        _original_print(*args, **kwargs)
        # Also write to log file
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                from datetime import datetime
                msg = ' '.join(str(arg) for arg in args)
                if msg.strip():
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [INFO] {msg}\n")
                    f.flush()
        except Exception:
            pass
    
    # Replace print temporarily for server thread
    builtins.print = _logged_print
    
    # Run the server in a thread so pywebview can take control of main thread
    def run_server():
        print(f"Starting HeartMuLa Studio server...")
        print(f"Logs directory: {logs_dir}")
        print(f"Console log file: {log_file_path}")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True
        )
    
    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()
    
    # Wait for server to be ready
    wait_for_server()
    
    # Launch pywebview window
    try:
        import webview
        print("Opening HeartMuLa Studio window...")
        
        # Create window with custom settings - window object not needed
        webview.create_window(
            'HeartMuLa Studio',
            'http://127.0.0.1:8000',
            width=1400,
            height=900,
            resizable=True,
            fullscreen=False,
            min_size=(800, 600),
            background_color='#1a1a1a',
            text_select=True,
            on_top=False,  # Keep in foreground but not always on top
            focus=True     # Get focus on creation
        )
        
        # Register cleanup handler for graceful shutdown
        atexit.register(cleanup)
        
        # Start the webview - this blocks until window is closed
        # When window closes, this returns and the program continues to exit
        webview.start(gui='cocoa')  # Explicitly use Cocoa for macOS
        
        # Window has been closed by user
        print("\nWindow closed by user")
        
    except ImportError:
        print("Warning: pywebview not available, falling back to browser")
        import webbrowser
        webbrowser.open("http://127.0.0.1:8000")
        # Keep the server running
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Error launching window: {e}")
        print("Falling back to browser...")
        import webbrowser
        webbrowser.open("http://127.0.0.1:8000")
        # Keep the server running
        while True:
            time.sleep(1)

def main():
    """Main entry point."""
    global _lock_socket
    
    try:
        # Check if another instance is already running
        _lock_socket = check_single_instance()
        if _lock_socket is None:
            print("Another instance of HeartMuLa Studio is already running.")
            print("Only one instance can be opened at a time.")
            sys.exit(0)
        
        app_dir, logs_dir = setup_environment()
        launch_server(app_dir, logs_dir)
        
        # If we reach here, the window was closed gracefully
        # cleanup() will be called by atexit
        sys.exit(0)
        
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
