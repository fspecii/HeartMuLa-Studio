#!/bin/bash
# HeartMuLa Studio Launcher
# Double-click this file to launch HeartMuLa Studio

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Launch the app
echo "Starting HeartMuLa Studio..."
open -a "HeartMuLa.app"

# Keep the terminal open briefly to show any errors
sleep 2
