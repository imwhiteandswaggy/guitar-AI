# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Clone the Repository

```bash
git clone https://github.com/imwhiteandswaggy/guitar-AI.git
cd guitar-AI
```

## Step 2: Run Setup Script

**Windows:**
```bash
setup.bat
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Download model file

## Step 3: Run the App

**Windows:**
```bash
venv\Scripts\activate
python app.py
```

**Mac/Linux:**
```bash
source venv/bin/activate
python app.py
```

## Step 4: Open in Browser

```
http://localhost:5000
```

## Step 5: Start Learning!

1. **Allow camera access** when prompted
2. **Point camera at your guitar** neck
3. **Select a chord** from the chord trainer
4. **Follow the overlay** to place your fingers
5. **See real-time feedback** as you play

## Troubleshooting

### Camera Not Working?
- Check camera permissions (especially Mac)
- Try different camera from dropdown
- Close other apps using camera

### Audio Not Working?
- Run: `python test_audio_devices.py`
- Check microphone permissions
- Ensure microphone is not muted

### Model File Missing?
- Run: `python download_model.py`
- Or download manually from GitHub Releases

### Still Having Issues?

Check the full [README.md](README.md) for detailed troubleshooting.

---

**That's it! You're ready to learn guitar with AI! 🎸**
