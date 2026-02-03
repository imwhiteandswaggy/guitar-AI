# 🎸 Guitar Teacher AI - Learn Guitar with AI

An AI-powered guitar learning app that uses computer vision and audio detection to provide real-time feedback on your playing. Point your camera at your guitar and see chord overlays, note detection, and finger position guidance in real-time.

![Guitar Teacher AI](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **🎯 Real-Time Chord Overlay** - AR-style chord diagrams projected directly on your guitar
- **🎵 Audio Detection** - Real-time pitch detection with 99% accuracy
- **👋 Hand Tracking** - MediaPipe hand tracking for finger position detection
- **🎸 Visual Feedback** - See exactly which strings and frets you're pressing
- **📱 Universal Camera Support** - Works with webcam, iPhone, Android, or any camera
- **🌐 Web-Based UI** - Modern, responsive interface accessible from any device
- **🎨 Professional Design** - Clean, polished UI with smooth animations

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.9+ recommended)
- **Webcam or camera-enabled device**
- **Guitar** (acoustic or electric)
- **Windows/Mac/Linux** (all supported)

### Installation

#### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
# Download and run setup script
setup.bat
```

**Mac/Linux:**
```bash
# Make script executable and run
chmod +x setup.sh
./setup.sh
```

#### Option 2: Manual Setup

1. **Clone the repository:**
```bash
git clone https://github.com/imwhiteandswaggy/guitar-AI.git
cd guitar-AI
```

2. **Create virtual environment (recommended):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download model file:**
```bash
# The model file is too large for GitHub
# Run the download script:
python download_model.py

# Or manually download from:
# https://github.com/imwhiteandswaggy/guitar-AI/releases/download/v1.0/best.pt
# Place it in: trained_models/real_guitar_test3/weights/best.pt
```

5. **Run the app:**
```bash
python app.py
```

6. **Open in browser:**
```
http://localhost:5000
```

## 📖 Usage

### Web Interface

1. **Start the app** - Run `python app.py`
2. **Open browser** - Navigate to `http://localhost:5000`
3. **Allow camera access** - Grant permissions when prompted
4. **Position your guitar** - Point camera at guitar neck
5. **Start learning!** - Select a chord and follow the overlay

### Controls

- **Camera Selector** - Choose your camera from dropdown
- **Free Play Mode** - See detected notes in real-time
- **Chord Trainer Mode** - Learn chords with visual guidance
- **Overlay Toggle** - Show/hide chord overlays
- **Calibrate** - Manually calibrate string positions for accuracy

### Keyboard Shortcuts (Desktop)

- **C** - Cycle through chords
- **O** - Toggle overlay
- **K** - Manual string calibration
- **R** - Reset calibration

## 🛠️ Troubleshooting

### Camera Not Working

**Problem:** Camera doesn't open or shows black screen

**Solutions:**
1. Check camera permissions (especially on Mac)
2. Try different camera from dropdown
3. Close other apps using camera
4. Restart the app

**Mac specific:**
```bash
# Grant camera permissions
# System Preferences > Security & Privacy > Camera
```

**Linux specific:**
```bash
# Install v4l2 backend
sudo apt-get install v4l-utils
```

### Audio Not Working

**Problem:** No audio detection

**Solutions:**
1. Check microphone permissions
2. Run audio device tester:
   ```bash
   python test_audio_devices.py
   ```
3. Select correct audio device
4. Check microphone is not muted

### Model File Missing

**Problem:** `FileNotFoundError: trained_models/real_guitar_test3/weights/best.pt`

**Solution:**
```bash
# Download model
python download_model.py

# Or manually:
# 1. Go to: https://github.com/imwhiteandswaggy/guitar-AI/releases
# 2. Download best.pt
# 3. Place in: trained_models/real_guitar_test3/weights/
```

### Dependencies Installation Issues

**Problem:** `pip install` fails

**Solutions:**

**Windows:**
```bash
# Install Visual C++ Build Tools if needed
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**Mac:**
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

**Linux:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip libopencv-dev
```

### Port Already in Use

**Problem:** `Address already in use`

**Solution:**
```bash
# Change port in app.py (last line)
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

## 📁 Project Structure

```
guitar-AI/
├── app.py                          # Main Flask web app
├── guitar_teacher_*.py             # Standalone versions
├── chord_library.py                # Chord definitions
├── chord_overlay.py                # Overlay rendering
├── string_refinement.py            # String position refinement
├── string_tracking.py              # Temporal smoothing
├── string_calibration.py           # Calibration utilities
├── requirements.txt                # Python dependencies
├── setup.sh / setup.bat            # Setup scripts
├── download_model.py               # Model downloader
├── templates/
│   └── index.html                  # Web UI
├── static/
│   ├── css/
│   │   └── styles.css              # Styles
│   └── js/
│       └── app.js                  # Frontend logic
└── trained_models/
    └── real_guitar_test3/
        └── weights/
            └── best.pt              # YOLOv8 model (download separately)
```

## 🔧 Configuration

### Camera Settings

Edit `app.py` to change camera defaults:
```python
# Line ~380
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Height
camera.set(cv2.CAP_PROP_FPS, 30)             # FPS
```

### Audio Settings

Audio device is auto-detected. To use specific device:
```python
# Edit app.py line ~172
self.audio.start(device=YOUR_DEVICE_ID)  # Get ID from test_audio_devices.py
```

### Model Path

If model is in different location:
```python
# Edit app.py line ~26
FRET_MODEL = "path/to/your/model.pt"
```

## 🎯 How It Works

### Computer Vision Pipeline

1. **YOLOv8 Detection** - Detects frets, neck, and nut (90%+ accuracy)
2. **String Calculation** - Geometric calculation with tapered spacing
3. **Edge Refinement** - Edge detection refines string positions
4. **Temporal Smoothing** - EMA filtering prevents glitchy detection
5. **Hand Tracking** - MediaPipe tracks finger positions
6. **Position Mapping** - Maps fingers to strings and frets

### Audio Pipeline

1. **Real-time Capture** - Records from default microphone
2. **Pitch Detection** - Librosa piptrack for frequency analysis
3. **Note Conversion** - Converts frequency to musical note
4. **Cross-Validation** - Validates visual predictions

### Overlay System

1. **Chord Selection** - User selects chord to learn
2. **Position Calculation** - Converts (string, fret) to pixel coordinates
3. **Perspective Correction** - Adjusts for camera angle
4. **Rendering** - Draws dots, finger numbers, and note names
5. **Feedback** - Shows green when fingers match positions

## 🧪 Testing

### Test Audio Devices
```bash
python test_audio_devices.py
```

### Test String Detection
```bash
python test_string_detection.py
```

### Test Model Loading
```python
from ultralytics import YOLO
model = YOLO("trained_models/real_guitar_test3/weights/best.pt")
print("✓ Model loaded successfully")
```

## 📊 Performance

- **Detection Speed**: 20-30 FPS on CPU, 60+ FPS on GPU
- **Accuracy**: 
  - Fret detection: 90.4% mAP
  - Neck detection: 97.6% mAP
  - String positions: 85-95% (with calibration)
- **Latency**: <100ms end-to-end

## 🌍 Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows 10/11 | ✅ Full | Tested on Windows 10/11 |
| macOS | ✅ Full | Requires camera permissions |
| Linux | ✅ Full | May need v4l2 backend |
| iPhone Safari | ✅ Partial | Camera works, some features limited |
| Android Chrome | ✅ Partial | Camera works, some features limited |

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Improvement

- [ ] Support for different tunings
- [ ] More chord variations
- [ ] Song mode (chord progressions)
- [ ] Recording and playback
- [ ] Mobile app version
- [ ] Better lighting compensation
- [ ] GPU acceleration options

## 📝 License

MIT License - feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **MediaPipe** by Google
- **Librosa** for audio processing
- **Flask** for web framework
- Training data from Roboflow guitar datasets

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/imwhiteandswaggy/guitar-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/imwhiteandswaggy/guitar-AI/discussions)

## 🎓 Learning Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [Librosa Tutorial](https://librosa.org/doc/latest/tutorial.html)

---

**Built with ❤️ for guitar learners everywhere**

*Last updated: January 2026*
