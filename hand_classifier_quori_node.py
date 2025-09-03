#!/usr/bin/env python3
import os
import subprocess
import tempfile
import cv2, pickle
import numpy as np
from openai import OpenAI
import detect_hand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import rospy
from std_msgs.msg import String
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import whisper
from geometry_msgs.msg import *
from geometry_msgs.msg import Twist
from sound_play.msg import SoundRequest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# You must add this line:
from cv_bridge import CvBridge

# And this line to use the ROS image message type:
from sensor_msgs.msg import Image


#Zane reverse priority code ---
class ReversePriorityQueue(object):
    def __init__(self):
        self.queue = []

    def insert(self, data):
        self.queue.append(data)

    def delete(self):
        min_idx = 0
        for i in range(len(self.queue)):
            if self.queue[i] < self.queue[min_idx]:
                min_idx = i
        item = self.queue[min_idx]
        del self.queue[min_idx]
        return item

class HandClassifier:
    def __init__(self, data_file):
        # node init is handled in main()
        self.detector = detect_hand.HandDetector(num_hands=1)
        self.last_stable_label = ""
        #self.sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        #self.img_pub = rospy.Publisher("/camera/rgb/image_blurred" , Image, queue_size=1)
        self.traj_pub = rospy.Publisher("/quori/joint_trajectory_controller/command", JointTrajectory, queue_size=10)
        #self.speech_pub = rospy.Publisher("/robotsound", SoundRequest, queue_size=10)
        self.model=whisper.load_model("base")
        with open(data_file, 'rb') as file:
            self.angles, self.labels = pickle.load(file)
        self.gen_initial_embeddings()
        # Load Whisper model for offline/local transcription
        try:
            rospy.loginfo("Loading Whisper model (this may take a while)...")
            self.whisper_model = whisper.load_model(rospy.get_param('~whisper_model', "base"))
            rospy.loginfo("Whisper model loaded")
        except Exception as e:
            rospy.logwarn(f"Failed to load Whisper model: {e}")
            self.whisper_model = None

    def gen_initial_embeddings(self, n_components=7):
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.angles)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(scaled_data)
        self.embeddings = self.pca.transform(scaled_data)
    bridge = CvBridge()
    def dist(self, a, b):
        return np.sum(np.square(a - b))

    def get_label(self, embedding, k=5):
        rpq = ReversePriorityQueue()
        for i, emb in enumerate(self.embeddings):
            rpq.insert((self.dist(emb, embedding), self.labels[i]))

        neighbors = {}
        distances = []
        for _ in range(k):
            distance, label = rpq.delete()
            distances.append(distance)
            neighbors[label] = neighbors.get(label, 0) + 1

        max_count = 0
        label = None
        for k, v in neighbors.items():
            if v > max_count:
                max_count = v
                label = k
        return label, distances

    def transcribe_file(self, audio_path: str, language: str = None) -> str:
        """Transcribe a local audio file using the loaded whisper model."""
        if self.whisper_model is None:
            rospy.logwarn("Whisper model not available")
            return ""
        try:
            opts = {}
            if language:
                opts["language"] = language
            result = self.whisper_model.transcribe(audio_path, **opts)
            text = result.get("text", "")
            rospy.loginfo(f"Whisper transcription: {text}")
            return text
        except Exception as e:
            rospy.logwarn(f"Whisper transcription failed: {e}")
            return ""

    def classify(self, im, is_bgr=True, display=False, stable_threshold=10.0):
        hands_found = self.detector.detect(im, is_bgr)
        if not getattr(hands_found, "multi_hand_landmarks", None):
            # No hands detected -> return None with infinite distance
            return "None", np.inf

        # Evaluate all hands and pick the candidate with smallest mean distance
        best_candidate = "None"
        best_mean_distance = np.inf
        for hand in hands_found.multi_hand_landmarks:
            angle_vec = detect_hand.generate_angle_vector(hand)
            embedding = self.pca.transform(self.scaler.transform(angle_vec.reshape(1, -1)))
            candidate_label, dists = self.get_label(embedding)
            candidate_mean = float(np.mean(dists))
            if candidate_mean < best_mean_distance:
                best_mean_distance = candidate_mean
                best_candidate = candidate_label

        # Stability logic: only commit to the new label if it's below the threshold,
        # otherwise keep the last stable label (or "None" if none yet).
        if best_candidate == self.last_stable_label:
            out_label = best_candidate
        else:
            if best_mean_distance <= stable_threshold:
                self.last_stable_label = best_candidate
                out_label = best_candidate
            else:
                out_label = self.last_stable_label if self.last_stable_label else "None"

        if display:
            detect_hand.draw_landmarks_on_image(im, hands_found)

        return out_label, best_mean_distance

# --- Quori gestures: wave, "fist", "woohoo-esque"
wave_positions = [[0.0, 0.0], [0.5, 0.2], [0.5, -0.2], [0.5, 0.2], [0.0, 0.0]]
fist_pos = [1.22, 0.0]
both_hands_pos = [0.5, 0.5]  # example for peace/ok both arms



#   /quori/arm_<side>/cmd_pos_dir   (Vector3: x=outer motor rad, y=inner motor rad, z=time or -1)
#   /quori/arm_<side>/shoulder_pos  (Vector3: x=J1 rad, y=J2 rad, z=unused)
arm_mcmd_pub = None
arm_joint_pub = None
traj_pub = None   # module-level fallback JointTrajectory publisher (set in main)
image_pub = rospy.Publisher("/hand_image_blurred", Image, queue_size=1)


from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def move_arm(joint_pub, joint_angles, duration=2.0):
    """
    Publish a single-point JointTrajectory using the provided joint_pub.
    If joint_pub is None, fall back to the module-level traj_pub.
    """
    pub = joint_pub if joint_pub is not None else traj_pub
    if pub is None:
        rospy.logwarn("No JointTrajectory publisher available to move arm")
        return
    try:
        traj = JointTrajectory()
        traj.joint_names = ["l_shoulder_pitch", "l_shoulder_roll",
                    "r_shoulder_pitch", "r_shoulder_roll", "waist_pitch"]

        point = JointTrajectoryPoint()
        point.positions = list(joint_angles)
        point.time_from_start = rospy.Duration(duration)
        traj.points.append(point)
        pub.publish(traj)
        rospy.loginfo(f"Published JointTrajectory: {joint_angles}")
    except Exception as e:
        rospy.logwarn(f"move_arm failed: {e}")

def perform_wave(pub):
    neutral = [0.0, -0.7, 0.0, -0.7, 0.0]
    # 5 joints: [l_shoulder_pitch, l_shoulder_roll, r_shoulder_pitch, r_shoulder_roll, waist_pitch]
    wave_steps = [
        neutral,      # all neutral
        [0.0, -0.7, 0.0, 0.7, 0.0],      # raise right arm
        [0.0, -0.7, -0.2, 0.7, 0.0],     # swing right arm
        [0.0, -0.7, 0.0, 0.7, 0.0],      # swing back
        neutral      # back to neutral
    ]
    for step in wave_steps:
        move_arm(pub, step, duration=0.4)
        rospy.sleep(0.4)



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import threading
def speak(message):
    def _tts():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                speech_file = f.name
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=message,
            ) as response:
                response.stream_to_file(speech_file)
            subprocess.call(["mpg123", "-q", speech_file])
        except Exception as e:
            rospy.logwarn(f"TTS failed: {e}")
    threading.Thread(target=_tts, daemon=True).start()

classifier = None
#not sure if i need these here
hand_pub = None
arm_pub = None
bridge = CvBridge()
meaningful_gestures = {"fist", "open_palm", "peace_sign", "ok_sign", "middle_finger"}

#cooldown state 
last_published_label = "None"
last_published_time = 0.0
COOLDOWN_PERIOD = 4.0

DETECTION_COUNTS = {}            
STABLE_FRAMES_REQUIRED = 3      
DIST_THRESHOLD = 6.0          # mean-distance threshold to accept gestures (was 5)

def image_callback(self, msg):
    global classifier, hand_pub, arm_pub, bridge, meaningful_gestures
    global last_published_label, last_published_time, COOLDOWN_PERIOD
    global DETECTION_COUNTS, STABLE_FRAMES_REQUIRED
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logerr(f"CvBridge error (bgr8): {e}")
        try:
            # If 'bgr8' fails, try 'rgb8' as a fallback.
            cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            rospy.logerr(f"CvBridge error (rgb8): {e}")
            return 

    label, dist = classifier.classify(cv_image, is_bgr=True, display=True)

    rospy.logdebug(f"[hand_classifier] candidate='{label}' mean_dist={dist}")

   
    for k in list(DETECTION_COUNTS.keys()):
        if k != label:
            DETECTION_COUNTS[k] = 0
    DETECTION_COUNTS[label] = DETECTION_COUNTS.get(label, 0) + 1

    now = time.time()
    should_publish = False

    rospy.logdebug(f"[hand_classifier] DETECTION_COUNTS={DETECTION_COUNTS} label={label} dist={dist:.2f} last_published={last_published_label}")

    # Only publish if candidate is meaningful, confident, and stable for N frames
    if dist <= DIST_THRESHOLD and label in meaningful_gestures:
        if DETECTION_COUNTS.get(label, 0) >= STABLE_FRAMES_REQUIRED:
            # still apply cooldown to avoid repeated triggers
            
                
            if label != last_published_label or (now - last_published_time) > COOLDOWN_PERIOD:
                should_publish = True

    if should_publish:
        hand_pub.publish(label)
        
        last_published_label = label
        last_published_time = now
        rospy.loginfo(f"Gesture accepted: {label} (dist={dist}) after {DETECTION_COUNTS[label]} frames")
        # Directly call the correct function for Quori's arm.
        if arm_pub is None:
            rospy.logwarn("arm_pub is not initialized, cannot move arms")
        else:
            
            if label == "peace_sign":
                perform_wave(arm_pub)
                speak("Peace dude!")
            elif label == "fist":
                perform_wave(arm_pub)
                speak("Ohhhhh yeah!")
            elif label == "ok_sign":
                perform_wave(arm_pub)
                speak("Oak E doh key!")
            elif label == "open_palm":
                perform_wave(arm_pub)
                speak("Hello!")
    else:
        hand_pub.publish("None")
        rospy.logdebug("No meaningful gesture published (unstable/confident)")
 
    cv2.putText(cv_image, f"{label} ({np.round(dist, 2)})", (50, 50),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if label == "middle_finger" and getattr(classifier.detector.detect(cv_image), "multi_hand_landmarks", None):
        hand_landmarks = classifier.detector.detect(cv_image).multi_hand_landmarks[0]
        x, y, w_box, h_box = bbox(hand_landmarks, cv_image.shape)
        roi = cv_image[y:y+h_box, x:x+w_box]
        blurred_roi = cv2.GaussianBlur(roi, (21, 21), 0)
        cv_image[y:y+h_box, x:x+w_box] = blurred_roi

	  


    # Display the image in a window and keep it updated.
    cv2.imshow("Quori Hand Classifier", cv_image)
    cv2.waitKey(1)
def bbox(hand_landmarks, img_shape, margin=20):
    h, w, _ = img_shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
    
    x_min = max(0, min(x_coords) - margin)
    x_max = min(w, max(x_coords) + margin)
    y_min = max(0, min(y_coords) - margin)
    y_max = min(h, max(y_coords) + margin)
    
    return x_min, y_min, x_max - x_min, y_max - y_min


def main():
    global classifier, hand_pub, arm_pub
    global arm_mcmd_pub, arm_joint_pub
    global trans_pub

    rospy.init_node('hand_classifier_quori_node', anonymous=True)

    # Publishers
    hand_pub = rospy.Publisher('/quori/hand_classifier/gesture', String, queue_size=10)
    # Publish JointTrajectory to the active controller (kept as fallback)
    arm_pub = rospy.Publisher('/quori/joint_trajectory_controller/command', JointTrajectory, queue_size=10)
    # Also create Quori-specific Vector3 publishers (preferred for motor-level commands).
    st = rospy.get_param('~arm_side', 'right')  # use 'right' or 'left' as appropriate
    arm_mcmd_pub = rospy.Publisher(f'/quori/arm_{st}/cmd_pos_dir', Vector3, queue_size=1)
    arm_joint_pub = rospy.Publisher(f'/quori/arm_{st}/shoulder_pos', Vector3, queue_size=1)
    rospy.sleep(1.0)  # allow publishers to register with master

    # Initialize classifier
    classifier = HandClassifier("hand_signs.pickle")
    # Publisher for transcripts
    trans_pub = rospy.Publisher('/whisper/transcript', String, queue_size=2)

    def _transcribe_path_cb(msg):
        if classifier is None:
            rospy.logwarn("Received transcribe request but classifier not initialized")
            return
        path = msg.data
        rospy.loginfo(f"Transcribe request: {path}")
        transcript = classifier.transcribe_file(path)
        trans_pub.publish(transcript)

    rospy.Subscriber('/whisper/transcribe_path', String, _transcribe_path_cb)

    # Image topic (adjust if needed)
    image_topic = "/astra_ros/devices/default/color/image_color"
    rospy.Subscriber(image_topic, Image, image_callback)

    rospy.loginfo("Hand classifier node started. Waiting for images...")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
