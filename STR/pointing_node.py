#!/usr/bin/env python3

from classify_hand import HandClassifier
from vlm_basic import VisionToText
import cv2
import numpy as np
import rospy
from std_msgs.msg import String 

rospy.init_node("pointing_vlm_node", anonymous = True)
speech_pub = rospy.Publisher("/quori/speech", String, queue_size=10)

# --- Initialize ---
classifier = HandClassifier("hand_signs.pickle")
vtt = VisionToText()

# Try different camera indices
for camera_index in range(10):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Using camera index: {camera_index}")
        break
    cap.release()
else:
    print("Error: Could not open any video capture.")
    exit()

    label, dist = classifier.classify(frame)

    if label == "point" and dist < 50:
        
        detected_hands = self.classifier.detector.detect(frame).multi_hand_landmarks
        print(f"Number of hands detected: {len(detected_hands) if detected_hands else 0}")
        if detected_hands:
            hand_landmarks = detected_hands[0]
            wrist = hand_landmarks.landmark[0]
            index_finger = hand_landmarks.landmark[8]

            h, w, _ = frame.shape

            wrist_pixel = (int(w * wrist.x), int(h * wrist.y))
            index_pixel = (int(w * index_finger.x), int(h * index_finger.y))
            
            scale = 1.0
            direction = (index_pixel[0] - wrist_pixel[0], index_pixel[1] - wrist_pixel[1])
            end_pixel = (int(index_pixel[0] + direction[0] * scale),
                         int(index_pixel[1] + direction[1] * scale))

            crop_size = 100
            x1 = max(0, end_pixel[0] - crop_size // 2)
            y1 = max(0, end_pixel[1] - crop_size // 2)
            x2 = min(w, end_pixel[0] + crop_size // 2)
            y2 = min(h, end_pixel[1] + crop_size // 2)

            if x2 > x1 and y2 > y1:
                centered_crop = frame[y1:y2, x1:x2]
                cv2.imshow("centered object", centered_crop)
                cv2.imwrite("pointed_obj_img.png", centered_crop)
                
                if centered_crop.size > 0:
                    description = vtt.viz_to_text(img=centered_crop, prompt="What object is this?")
                    
                    # pubslish the speaking
                    msg = String()
                    msg.data = description
                    speech_pub.publish(msg)
                    
                    timestamp = rospy.Time.now()
                    with open("vlm_log.txt", "a") as f:
                        f.write(f"{timestamp}: {description}\n")
                    print("VLM sees:", description)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, description[:30], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                print("Invalid crop dimensions.")

    cv2.imshow("Frame", frame)
    test_crop = frame[50:150, 50:150]
    cv2.imshow("test crop", test_crop)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()
