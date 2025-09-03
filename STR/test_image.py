import cv2
import numpy as np
from classify_hand import HandClassifier
from vlm_basic import VisionToText

# --- Initialize ---
classifier = HandClassifier("hand_signs.pickle")
vtt = VisionToText()

# --- Load your test image ---
image_path = "tester.jpg"
frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
else:
    print("image loaded")

scale_percent = 640/frame.shape[1]
new_width = int(frame.shape[1]*scale_percent)
new_height = int(frame.shape[0]*scale_percent)
frame = cv2.resize(frame, (new_width, new_height))

# --- Detect hand gesture ---
label, dist = classifier.classify(frame, display=True)
print("Detected label:", label, "Distance:", dist)

# Add a check for detected hands before accessing the list
detected_hands = classifier.detector.detect(frame).multi_hand_landmarks

if detected_hands and label == "point":
    hands = detected_hands[0]
   
    # Get wrist and index finger
    wrist = hands.landmark[0]
    index = hands.landmark[8]

    h, w, _ = frame.shape
    wrist_px = (int(wrist.x*w), int(wrist.y*h))
    index_px = (int(index.x*w), int(index.y*h))

    # Draw pointing line
    cv2.line(frame, wrist_px, index_px, (0,255,0), 2)

    # Extend pointing vector
    scale = 0.5
    end_px = (int(index_px[0] + (index_px[0]-wrist_px[0])*scale),
              int(index_px[1] + (index_px[1]-wrist_px[1])*scale))

    # Crop around pointed location
    crop_size = 100
    x1 = max(0, end_px[0] - crop_size // 2)
    y1 = max(0, end_px[1] - crop_size // 2)

    x2 = min(w, end_px[0] + crop_size // 2)
    y2 = min(h, end_px[1] + crop_size // 2)
        
    print(f"end_px: {end_px}")
    print(f"w: {w}, h: {h}")
    print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

    # --- Check for valid crop and process ---
    if x2 > x1 and y2 > y1: 
        pointed_crop = frame[y1:y2, x1:x2]

        if pointed_crop.size > 0:
            description = vtt.viz_to_text(img=pointed_crop, prompt="What object is this?")
            print("VLM sees:", description)
            cv2.imshow("Pointed Crop", pointed_crop)
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, description[:30], (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            print("Error: pointed_crop is empty or invalid.")
            print("VLM sees: No object detected at point")
            
    else:
        print("Crop dimensions are invalid.")
        print("VLM sees: No object detected at point")

else:
    print("No pointing hand detected.")

cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()