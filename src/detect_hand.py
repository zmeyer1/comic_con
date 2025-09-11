# Theft from
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=_JVO3rvPD4RN
# Make sure to grab the task descriptor: 
#    wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

# indices of the hand detector points that compose the segments of the hand
hand_segments = [
    (0,1), # wrist to thumb 0 
    (1,2), # thumb 1
    (2,3), # thumb 2
    (3,4), # thumb 3
    (0,5), # wrist to index 0
    (5,6), # index 1
    (6,7), # index 2
    (7,8), # index 3
    (0,9), # wrist to middle 0
    (9,10), # middle 1
    (10,11), # middle 2
    (11,12), # middle 3
    (0,13), # wrist to ring 0
    (13,14), # ring 1
    (14,15), # ring 2
    (15,16), # ring 3
    (0,17), # wrist to pinky 0
    (17,18), # pinky 1
    (18,19), # pinky 2
    (19,20), # pinky 3
    (1,13), # palm span
]
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDetector:
    # Little wrapper class for creating a hand detector
    def __init__(self, num_hands=1):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=num_hands)
        # self.detector = vision.HandLandmarker.create_from_options(options)
        self.detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)


    def detect(self, img, is_bgr=True):
        copy_img = img.copy()
        if is_bgr:
            copy_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB)
        return self.detector.process(copy_img)\

    def get_positions(self, img):
        # returns the indices of the hand landmarks in the image
        results = self.detect(img)
        if not results.multi_hand_landmarks:
            return None
        for hand in results.multi_hand_landmarks:
            landmark_list = hand.landmark
            wrist = landmark_list[0]
            # get the indices in the image
            i = int(wrist.x * img.shape[1])
            j = int(wrist.y * img.shape[0])
        return i,j

def generate_angle_vector(landmark_list, segments = hand_segments, confidence_threshold=0.0):
    # generates a vector of angles constructed from given segments of importance
    angles = []
    landmark_list = landmark_list.landmark
    for segment in segments:
        a = landmark_list[segment[0]]
        b = landmark_list[segment[1]]
        if a.visibility < confidence_threshold or b.visibility < confidence_threshold:
            angles.append(2*np.pi)
            angles.append(2*np.pi) # with pca, this probably really screws with data
            continue
        section = np.array([b.x,b.y,b.z]) - np.array([a.x,a.y,a.z])
        angles.append(np.arctan2(section[1], section[0]))
        angles.append(np.arctan2(section[2], section[0]**2 + section[1]**2))
    return np.array(angles)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green display color

def draw_landmarks_on_image(rgb_image, detection_result):
#   hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
    annotated_image = rgb_image

    if detection_result.multi_hand_landmarks:
        for hand_landmarks in detection_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #   # Loop through the detected hands to visualize.
    #   for idx in range(len(hand_landmarks_list)):
    #     hand_landmarks = hand_landmarks_list[idx]
    #     handedness = handedness_list[idx]

    #     # Draw the hand landmarks.
    #     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #     hand_landmarks_proto.landmark.extend([
    #       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    #     ])
    #     solutions.drawing_utils.draw_landmarks(
    #       annotated_image,
    #       hand_landmarks_proto,
    #       solutions.hands.HAND_CONNECTIONS,
    #       solutions.drawing_styles.get_default_hand_landmarks_style(),
    #       solutions.drawing_styles.get_default_hand_connections_style())

    #     # Get the top left corner of the detected hand's bounding box.
    #     height, width, _ = annotated_image.shape
    #     x_coordinates = [landmark.x for landmark in hand_landmarks]
    #     y_coordinates = [landmark.y for landmark in hand_landmarks]
    #     text_x = int(min(x_coordinates) * width)
    #     text_y = int(min(y_coordinates) * height) - MARGIN

    #     # Draw handedness (left or right hand) on the image.
    #     cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to Gray
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for display

        detection_result = detector.detect(frame)

        annotated_image = draw_landmarks_on_image(frame, detection_result, )
        cv2.imshow('window', annotated_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
