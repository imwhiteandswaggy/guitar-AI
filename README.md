# Guitar Teacher - AI-Powered Real-Time Guitar Learning App

An AI guitar teaching system that combines computer vision and audio detection to provide real-time feedback on your playing.

## Demo

Point your webcam at your guitar and the app will:
- Detect frets, neck, and finger positions using YOLOv8 object detection
- Track your hand movements with MediaPipe
- Listen to the notes you're playing with real-time audio pitch detection
- Cross-validate vision predictions with audio for 99% accuracy
- Show you exactly what notes you're playing in real-time

## Features

- **Visual Detection (90%+ accuracy)**
  - Fret detection: 90.4% mAP
  - Neck detection: 97.6% mAP
  - Geometric string position calculation: ~85-90% accuracy
  
- **Audio Detection (99% accuracy)**
  - Real-time pitch detection using librosa
  - Cross-validates visual predictions
  - Works with any webcam microphone

- **Hand Tracking**
  - MediaPipe hand tracking for finger positions
  - Shows which strings and frets you're pressing
  - Visual feedback with note names

- **Real-Time Performance**
  - 20-25 FPS on CPU
  - No calibration required
  - Plug-and-play experience

## Requirements

- Python 3.11+
- Webcam
- Guitar (acoustic or electric)
- Windows/Mac/Linux

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/guitar-teacher.git
cd guitar-teacher
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python guitar_teacher_audio_visual.py
```

## Usage

### Basic Controls
- **D** - Toggle debug mode (shows detection boxes)
- **H** - Toggle hand skeleton overlay
- **Q** - Quit

### First Time Setup

If the audio isn't working with your webcam:

1. Run the audio device tester:
```bash
python test_audio_devices.py
```

2. Find your webcam microphone in the list (usually contains "camera" or "webcam" in the name)

3. Edit `guitar_teacher_audio_visual.py` line 117 and add your device number:
```python
self.stream = sd.InputStream(
    device=2,  # ← Your webcam device number
    callback=self.audio_callback,
    ...
)
```

## How It Works

### Computer Vision Pipeline
1. **YOLOv8 Object Detection** - Detects frets and neck boundaries
2. **Geometric Calculation** - Calculates string positions from neck box
3. **MediaPipe Hand Tracking** - Tracks finger positions in 3D space
4. **Position Mapping** - Maps fingers to strings and frets

### Audio Pipeline
1. **Real-time Audio Capture** - Records from webcam microphone
2. **Pitch Detection** - Uses librosa's piptrack for frequency analysis
3. **Note Conversion** - Converts frequency to musical note
4. **Cross-Validation** - Validates visual predictions with audio

### Why Audio + Vision?

Vision alone achieves ~85% accuracy (strings are thin and hard to detect). Audio achieves 99% accuracy but doesn't show finger positions. Combining both gives you:
- Accurate note detection (from audio)
- Visual feedback on technique (from vision)
- Cross-validation when they disagree

## Project Structure

```
guitar-teacher/
├── guitar_teacher_audio_visual.py  # Main app with audio + vision
├── guitar_teacher_minimal.py       # Vision-only version (cleaner UI)
├── guitar_teacher_geometric.py     # Vision with geometric strings
├── test_audio_devices.py           # Audio device selector utility
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── trained_models/
    └── real_guitar_test3/
        └── weights/
            └── best.pt             # Trained YOLOv8 model (90% mAP)
```

## Model Training Details

The YOLOv8 model was trained on:
- **926 images** of real guitars from various angles
- **Classes**: fret, neck, nut
- **Training time**: ~45 minutes on RTX 4070 Ti
- **Final performance**:
  - Fret: 90.4% mAP50
  - Neck: 97.6% mAP50
  - Nut: 97.0% mAP50

String detection uses geometric calculation instead of ML (see [Technical Decisions](#technical-decisions) below).

## Technical Decisions

### Why Not ML for String Detection?

Initially attempted to train string detection but achieved only 44.8% mAP with 213 labeled images. Strings are:
- Very thin (1-2mm on camera)
- Low contrast with fretboard
- Easily occluded by hands
- Would require 1,000-5,000 images for 70-80% accuracy

**Solution**: Calculate string positions geometrically from the neck bounding box. Guitars have evenly-spaced strings by design, so this achieves 85-90% accuracy with no additional training data.

### Why Add Audio?

Audio detection is:
- 99% accurate (vs 85% for vision)
- Immune to occlusion
- Computationally lightweight
- Works in all lighting conditions

The combination provides the best user experience: audio gives ground truth, vision shows technique.

## Contributing

Pull requests welcome! Areas for improvement:
- [ ] UI/UX enhancements
- [ ] Support for different tunings
- [ ] Chord recognition
- [ ] Practice mode with feedback
- [ ] Recording and playback
- [ ] Mobile app version
- [ ] Better lighting compensation
- [ ] GPU acceleration option

## Known Limitations

- Vision accuracy degrades in poor lighting
- Hand occlusion can affect string detection
- Requires clear view of guitar neck
- Audio requires relatively quiet environment
- Currently only supports standard tuning (EADGBE)

## Performance Tips

- Ensure good lighting on the guitar neck
- Position camera to see at least 5-6 frets clearly
- Use a quiet room or good microphone for audio
- Run on GPU for 60+ FPS (add `device='cuda'` in model loading)

## License

MIT License - feel free to use, modify, and distribute.

## Acknowledgments

- YOLOv8 by Ultralytics
- MediaPipe by Google
- Librosa for audio processing
- Training data from Roboflow guitar datasets

## Contact

Questions? Issues? Feature requests? Open an issue or PR!

---

Built as a learning project to explore computer vision, audio processing, and multi-modal AI systems.
