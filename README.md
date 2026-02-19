# Guitar Teacher AI

I built this because I wanted a guitar coach that actually watches you play—like having someone sit across from you and tell you where to put your fingers. Point your camera at your guitar, pick a chord, and it overlays the finger positions right on the neck in real time. It also listens to what you're playing so you get both visual and audio feedback.

## What it does

- **Chord overlay** – AR-style dots and finger numbers projected onto your guitar neck
- **Hand tracking** – Uses your camera to see where your fingers are
- **Audio detection** – Picks up the notes you're playing and checks if they match
- **Chord trainer** – Pick a chord, play it, and get a score on how well you're hitting the right spots
- **Works with any camera** – Webcam, phone, whatever you've got

## Quick start

You'll need Python 3.8+, a camera, and a guitar.

**Windows:** Run `setup.bat`  
**Mac/Linux:** Run `chmod +x setup.sh` then `./setup.sh`

That'll set up the venv, install dependencies, and download the model. Then:

```bash
python app.py
```

Open http://localhost:5000 in your browser. Let it use your camera, point it at your guitar, and you're good to go.

**Manual setup** if the scripts don't work for you:

```bash
git clone https://github.com/winfieldhunter/guitar-AI.git
cd guitar-AI
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python download_model.py   # Model's too big for GitHub, gotta download separately
python app.py
```

## Using it

1. Run the app and open it in your browser
2. Allow camera access
3. Point the camera at your guitar neck (you want the fretboard in view)
4. Use the chord selector to pick what you want to learn
5. The overlay shows where to put your fingers—try to match it

There's a **Debug** button if you want to see the raw detection (neck box, frets, string lines). **Calibrate** lets you manually click on each string if the auto-detection is off. **Raw** shows the camera feed with no processing.

## When things go wrong

**Camera's black or not working** – Close other apps using the camera (Zoom, Teams, etc.). Try a different camera from the dropdown. On Mac, check System Preferences > Security & Privacy > Camera.

**No audio detection** – Check mic permissions. The app auto-detects your default input, but if it's wrong you might need to dig into the code.

**"Model file not found"** – Run `python download_model.py`. The trained model is ~50MB so it's not in the repo.

**Port already in use** – The app will try 5000, then 5001, 5002... so you should be fine. If not, something else might be hogging ports.

**Strings look wrong** – Hit **Calibrate** and click on each string from top to bottom (thinnest to thickest). That usually fixes it.

## How it works (the short version)

I trained a YOLO model on a bunch of labeled guitar images to detect the neck, frets, and nut. Strings are trickier, pure ML was too noisy, so I use a geometric model (tapered spacing, wider at the nut) and refine it with edge detection and fret intersections. Hand tracking is MediaPipe. Audio is librosa for pitch detection. Everything runs in a Flask app that streams the video with overlays to your browser.

## Project structure

```
app.py              # Main web app
chord_library.py    # Chord fingerings
chord_overlay.py    # Draws the dots on the neck
string_*.py         # String position math, refinement, smoothing
templates/          # HTML
static/             # CSS, JS
trained_models/     # YOLO weights (download separately)
```

## Stuff I'd like to add someday

- Different tunings
- Chord progressions / song mode
- Recording and playback
- Better handling for weird lighting
- Maybe a proper mobile app

## Credits

Built with YOLOv8 (Ultralytics), MediaPipe, Librosa, and Flask. Training data from Roboflow guitar datasets.

---

If you run into issues, open one on [GitHub](https://github.com/winfieldhunter/guitar-AI/issues). Happy playing.
