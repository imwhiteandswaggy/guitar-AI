"""
Create GitHub release v1.0 and upload best.pt
Run this once you have the trained model file.

Requirements:
  - best.pt (default: trained_models/real_guitar_test3/weights/best.pt, or pass path)
  - GITHUB_TOKEN env var with repo scope (create at https://github.com/settings/tokens)

Usage:
  set GITHUB_TOKEN=your_token
  python create_release.py
  python create_release.py "c:\path\to\best.pt"
"""

import os
import sys
import json
import urllib.request
import urllib.error

REPO = "winfieldhunter/guitar-AI"
DEFAULT_MODEL = "trained_models/real_guitar_test3/weights/best.pt"


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[FAIL] Set GITHUB_TOKEN environment variable (repo scope)")
        sys.exit(1)

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    if not os.path.exists(model_path):
        print(f"[FAIL] Model not found: {model_path}")
        print("Train the model first, or place best.pt at the path above.")
        sys.exit(1)

    # Check if release already exists
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/tags/v1.0",
        headers={"Authorization": f"token {token}"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            release = json.loads(r.read().decode())
            release_id = release["id"]
            print("[OK] Release v1.0 already exists, uploading asset...")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"[FAIL] API error: {e}")
            sys.exit(1)
        release_id = None

    if release_id is None:
        # Create release
        data = json.dumps({
            "tag_name": "v1.0",
            "name": "v1.0",
            "body": "Initial release with trained YOLOv8 guitar neck/fret/nut model.",
        }).encode()
        req = urllib.request.Request(
            f"https://api.github.com/repos/{REPO}/releases",
            data=data,
            headers={
                "Authorization": f"token {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as r:
                release = json.loads(r.read().decode())
                release_id = release["id"]
                print("[OK] Created release v1.0")
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"[FAIL] Create release: {e.code} {body}")
            sys.exit(1)

    # Upload asset
    with open(model_path, "rb") as f:
        model_data = f.read()

    url = f"https://uploads.github.com/repos/{REPO}/releases/{release_id}/assets?name=best.pt"
    req = urllib.request.Request(
        url,
        data=model_data,
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            asset = json.loads(r.read().decode())
            print(f"[OK] Uploaded best.pt ({len(model_data) / 1024 / 1024:.1f} MB)")
            print(f"     {asset.get('browser_download_url', '')}")
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        if "already_exists" in body or e.code == 422:
            print("[OK] best.pt already attached to v1.0")
        else:
            print(f"[FAIL] Upload: {e.code} {body}")
            sys.exit(1)

    print()
    print("Done. Users can now run: python download_model.py")


if __name__ == "__main__":
    main()
