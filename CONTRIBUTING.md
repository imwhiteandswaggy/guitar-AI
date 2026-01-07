# Contributing to Guitar Teacher

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

1. Fork the repo
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/guitar-teacher.git
cd guitar-teacher
```

3. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a branch:
```bash
git checkout -b feature/your-feature-name
```

## Code Style

- Follow PEP 8
- Use descriptive variable names
- Add comments for complex logic
- Keep functions focused and small

## Areas for Contribution

### Easy Issues (Good First Contributions)
- [ ] UI improvements (colors, layouts, fonts)
- [ ] Add keyboard shortcuts
- [ ] Improve error messages
- [ ] Add configuration file support
- [ ] Better progress indicators

### Medium Issues
- [ ] Support for alternate tunings (Drop D, DADGAD, etc.)
- [ ] Add practice modes (scales, chord progressions)
- [ ] Recording and playback features
- [ ] Save/load user progress
- [ ] Add chord recognition

### Hard Issues
- [ ] Mobile app version (iOS/Android)
- [ ] Improve string detection accuracy
- [ ] GPU acceleration
- [ ] Multi-guitar support
- [ ] Real-time tablature generation

## Testing Your Changes

1. Test with your own guitar/webcam
2. Try different lighting conditions
3. Test with both acoustic and electric guitars if possible
4. Verify audio detection works
5. Check FPS doesn't drop significantly

## Submitting a PR

1. Make sure your code works
2. Update README if you added features
3. Commit with clear messages:
```bash
git commit -m "Add: support for Drop D tuning"
```

4. Push to your fork:
```bash
git push origin feature/your-feature-name
```

5. Open a PR on GitHub with:
   - Description of changes
   - Screenshots/videos if UI changed
   - Any testing you did

## Questions?

Open an issue or ask in the PR!
