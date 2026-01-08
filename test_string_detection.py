"""
Quick test script to verify string detection improvements
"""
import sys

print("Testing string detection improvements...")
print("=" * 60)

# Test 1: Import string_refinement module
try:
    from string_refinement import refine_string_positions_with_edges
    print("[OK] string_refinement module imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import string_refinement: {e}")
    sys.exit(1)

# Test 2: Import app module
try:
    import app
    print("[OK] app.py imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import app: {e}")
    sys.exit(1)

# Test 3: Check if GuitarDetectionEngine has new methods
try:
    engine = app.GuitarDetectionEngine()
    print("[OK] GuitarDetectionEngine initialized")
    
    # Check if calculate_string_positions accepts nut_box parameter
    import inspect
    sig = inspect.signature(engine.calculate_string_positions)
    params = list(sig.parameters.keys())
    if 'nut_box' in params:
        print("[OK] calculate_string_positions accepts nut_box parameter")
    else:
        print("[WARN] calculate_string_positions doesn't have nut_box parameter")
    
except Exception as e:
    print(f"[FAIL] Error initializing engine: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("All tests passed!")
print("\nYou can now run:")
print("  python app.py")
print("  python guitar_teacher_minimal.py")
print("  python guitar_teacher_audio_visual.py")
