# How to Set Up Your GitHub Repo

## Step 1: Create the Repo on GitHub

1. Go to https://github.com/new
2. Repository name: `guitar-teacher` (or whatever you want)
3. Description: "AI-powered real-time guitar teaching app using computer vision and audio detection"
4. Choose: **Public** (for portfolio visibility)
5. **DO NOT** check "Initialize with README" (we already have one)
6. Click "Create repository"

## Step 2: Prepare Your Local Files

In your `C:\CAI_Projects\Guitar_coach` folder, organize files like this:

```
Guitar_coach/
├── .gitignore                          ← Download from outputs
├── README.md                           ← Download from outputs
├── CONTRIBUTING.md                     ← Download from outputs
├── requirements.txt                    ← Download from outputs
├── guitar_teacher_audio_visual.py     ← Your main app
├── guitar_teacher_minimal.py          ← Alternative version
├── guitar_teacher_geometric.py        ← Alternative version
├── test_audio_devices.py              ← Audio utility
└── trained_models/
    └── real_guitar_test3/
        └── weights/
            └── best.pt                 ← Your trained model (~6MB)
```

## Step 3: Initialize Git and Push

Open terminal in `C:\CAI_Projects\Guitar_coach`:

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Guitar Teacher AI app with audio + vision"

# Add your GitHub repo as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/guitar-teacher.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Verify Upload

Go to your repo URL: `https://github.com/YOUR_USERNAME/guitar-teacher`

You should see:
- ✅ README with nice formatting
- ✅ All Python files
- ✅ trained_models folder with best.pt

## Step 5: Share With Your Friend

Send him: `https://github.com/YOUR_USERNAME/guitar-teacher`

He can then:
```bash
git clone https://github.com/YOUR_USERNAME/guitar-teacher.git
cd guitar-teacher
pip install -r requirements.txt
python guitar_teacher_audio_visual.py
```

## Important Notes

### If Your Model File is Too Big

GitHub has a 100MB file limit. If `best.pt` is larger:

**Option A: Use Git LFS (Large File Storage)**
```bash
# Install Git LFS
git lfs install

# Track the model file
git lfs track "trained_models/**/*.pt"

# Add .gitattributes
git add .gitattributes

# Now add and commit normally
git add trained_models/
git commit -m "Add trained model with LFS"
git push
```

**Option B: Host Model Externally**
- Upload `best.pt` to Google Drive
- Share link in README
- Users download separately

### Making PRs Easy for Your Friend

1. In your GitHub repo, go to **Settings** → **Collaborators**
2. Add your friend so he can push directly
3. OR he can fork it and make PRs

## Recommended GitHub Settings

Go to your repo → **Settings**:

1. **Features**: Enable Issues, Wiki if you want
2. **Topics**: Add tags like `computer-vision`, `yolo`, `guitar`, `ai`, `python`, `mediapipe`
3. **Social Preview**: Upload a screenshot of the app running

## Next Steps After Upload

1. **Add a demo video/GIF** to README
2. **Add screenshots** showing the UI
3. **Star your own repo** (why not)
4. **Share on LinkedIn** for visibility
5. **Add to your resume** under projects

---

That's it! Once pushed, send your friend the link and he's good to go!
