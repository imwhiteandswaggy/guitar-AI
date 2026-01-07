"""
Audio Device Selector
Lists all available audio input devices and lets you test them
"""

import sounddevice as sd
import numpy as np

print("="*60)
print("AVAILABLE AUDIO INPUT DEVICES")
print("="*60)

# List all devices
devices = sd.query_devices()

input_devices = []
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        input_devices.append((i, device))
        print(f"\nDevice {i}:")
        print(f"  Name: {device['name']}")
        print(f"  Channels: {device['max_input_channels']}")
        print(f"  Sample Rate: {device['default_samplerate']}")
        
        # Highlight if it looks like a webcam
        if any(keyword in device['name'].lower() for keyword in ['camera', 'webcam', 'usb', 'integrated']):
            print("  ⭐ LIKELY WEBCAM")

print("\n" + "="*60)
print(f"\nCurrent default input: Device {sd.default.device[0]}")
print(f"Name: {devices[sd.default.device[0]]['name']}")
print("="*60)

# Let user select device
print("\nWhich device number do you want to use for the guitar teacher?")
print("(Look for your webcam or microphone above)")
device_num = input("Enter device number: ").strip()

try:
    device_num = int(device_num)
    device = devices[device_num]
    
    print(f"\n✓ You selected: {device['name']}")
    print(f"\nAdd this to your guitar_teacher_audio_visual.py:")
    print(f"\nIn the AudioDetector.start() method, change:")
    print(f"  self.stream = sd.InputStream(")
    print(f"      device={device_num},  ← ADD THIS LINE")
    print(f"      callback=self.audio_callback,")
    print(f"      channels=1,")
    print(f"      ...)")
    
    # Test recording
    print(f"\n\nTesting device {device_num}...")
    print("Speak or play guitar for 3 seconds...")
    
    recording = sd.rec(int(3 * device['default_samplerate']), 
                      samplerate=int(device['default_samplerate']), 
                      channels=1, 
                      device=device_num)
    sd.wait()
    
    # Check if we got audio
    max_vol = np.max(np.abs(recording))
    print(f"✓ Recording complete!")
    print(f"Max volume: {max_vol:.4f}")
    
    if max_vol > 0.01:
        print("✓ Device is working! Got audio signal.")
    else:
        print("⚠ Very quiet or no audio detected. Try another device?")
        
except ValueError:
    print("Invalid device number")
except Exception as e:
    print(f"Error: {e}")
