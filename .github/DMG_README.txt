═══════════════════════════════════════════════════════════
    HeartMuLa Studio for macOS
═══════════════════════════════════════════════════════════

Thank you for downloading HeartMuLa Studio!

🎵 INSTALLATION
───────────────────────────────────────────────────────────

1. Drag "HeartMuLa.app" to the "Applications" folder
2. Double-click the app to launch
3. On first launch:
   - macOS may show a security warning
   - Go to System Preferences → Security & Privacy
   - Click "Open Anyway" to allow the app

📦 FIRST RUN
───────────────────────────────────────────────────────────

The app will automatically download AI models (~5GB) from
HuggingFace on first launch. This may take 10-30 minutes
depending on your internet connection.

All data is stored in your user Library folder:
~/Library/Application Support/HeartMuLa/
  ├── models/              (AI models, ~5GB)
  ├── generated_audio/     (Your generated music)
  ├── ref_audio/           (Reference audio uploads)
  └── jobs.db              (Song history database)

Logs are stored in:
~/Library/Logs/HeartMuLa/

💻 SYSTEM REQUIREMENTS
───────────────────────────────────────────────────────────

• macOS 10.13 (High Sierra) or later
• Apple Silicon (M1/M2/M3) or Intel Mac
• 10GB+ RAM
• 15GB+ free disk space (for models and generated music)

🎸 METAL GPU ACCELERATION
───────────────────────────────────────────────────────────

HeartMuLa Studio is optimized for Apple Metal GPUs:
• Apple Silicon: Native acceleration with Metal Performance Shaders
• Intel Macs: Metal support for compatible GPUs

🎶 FEATURES
───────────────────────────────────────────────────────────

✓ AI-powered music generation up to 4+ minutes
✓ Instrumental mode
✓ Style tags and seed control
✓ Reference audio style transfer (experimental)
✓ AI-generated lyrics
✓ Queue system for multiple generations
✓ Professional Spotify-inspired interface

📚 DOCUMENTATION
───────────────────────────────────────────────────────────

Full documentation and source code:
https://github.com/audiohacking/HeartMuLa-Studio

🆘 TROUBLESHOOTING
───────────────────────────────────────────────────────────

App won't open?
  → Right-click app → Open → Click "Open" in dialog

Models not downloading?
  → Check internet connection
  → Check disk space (need 15GB+ free)
  → View logs: ~/Library/Logs/HeartMuLa/

Slow generation?
  → First generation compiles kernels (1-2 min)
  → Subsequent generations are faster

Where is my data stored?
  → All data: ~/Library/Application Support/HeartMuLa/
  → Generated songs: ~/Library/Application Support/HeartMuLa/generated_audio/
  → The app bundle itself is read-only and contains no user data

Still having issues?
  → Open an issue on GitHub
  → Include log files from ~/Library/Logs/HeartMuLa/

📝 LICENSE
───────────────────────────────────────────────────────────

HeartMuLa Studio is open source (MIT License)
Built on HeartLib: https://github.com/HeartMuLa/heartlib

═══════════════════════════════════════════════════════════

Made with ❤️ for the open-source AI music community
