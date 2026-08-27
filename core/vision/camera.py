import asyncio
import base64
import cv2

def capture_photo_b64() -> str:
    """
    Capture one frame from the local MJPEG stream, encode as JPEG,
    and return as a Base64 string.
    
    This runs synchronously; wrap in run_in_executor for async usage.
    """
    try:
        # Capture from local MJPEG stream
        cap = cv2.VideoCapture("http://localhost:5000/mjpeg", cv2.CAP_ANY)
        if not cap.isOpened():
            print("[Camera] Failed to open MJPEG stream.")
            return None

        # Give it a tiny bit of time to grab a fresh frame
        import time
        time.sleep(0.1)

        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print("[Camera] Failed to capture frame from stream.")
            return None
            
        # Optional: Save locally for debugging/reference (off by default — SD-card
        # writes + latency on every capture)
        import os
        if os.environ.get("KIKI_DEBUG_SAVE_FRAME"):
            cv2.imwrite("image.jpeg", frame)

        # Encode as JPEG
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            print("[Camera] Failed to encode frame to JPEG.")
            return None
            
        img_bytes = buf.tobytes()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_b64
        
    except Exception as e:
        print(f"[Camera] Error capturing photo: {e}")
        return None
