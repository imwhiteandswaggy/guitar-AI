"""
Model Downloader
Downloads the trained YOLOv8 model file if not present
"""

import os
import urllib.request
import sys

MODEL_URL = "https://github.com/winfieldhunter/guitar-AI/releases/download/v1.0/best.pt"
MODEL_PATH = "trained_models/real_guitar_test3/weights/best.pt"

def download_file(url, destination):
    """Download file with progress bar"""
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '=' * filled + '-' * (bar_length - filled)
        sys.stdout.write(f'\r[{bar}] {percent:.1f}%')
        sys.stdout.flush()
    
    try:
        print(f"Downloading model from {url}...")
        print(f"Destination: {destination}")
        urllib.request.urlretrieve(url, destination, show_progress)
        print("\n[OK] Download complete!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Download failed: {e}")
        return False

def main():
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"[OK] Model already exists at {MODEL_PATH}")
        print("Skipping download.")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Download model
    print("="*60)
    print("Guitar Teacher AI - Model Downloader")
    print("="*60)
    print()
    
    success = download_file(MODEL_URL, MODEL_PATH)
    
    if success:
        print()
        print("="*60)
        print("[OK] Model downloaded successfully!")
        print("="*60)
        print(f"Model location: {MODEL_PATH}")
        print()
        print("You can now run the app:")
        print("  python app.py")
    else:
        print()
        print("="*60)
        print("[FAIL] Download failed")
        print("="*60)
        print()
        print("Alternative options:")
        print("1. Check your internet connection")
        print("2. Download manually from:")
        print(f"   {MODEL_URL}")
        print("3. Place the file at:")
        print(f"   {MODEL_PATH}")
        sys.exit(1)

if __name__ == "__main__":
    main()
