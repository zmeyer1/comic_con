#!/usr/bin/env python3
import cv2
import numpy as np
import os
from classify_hand import HandClassifier
import rospy
from std_msgs.msg import String
import time
from geometry_msgs.msg import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import threading, random
# from gtts import gTTS
import pyttsx3


# these could be made into ros params
STABLE_FRAMES_REQUIRED = 3      
DIST_THRESHOLD = 9.0          # mean-distance threshold to accept gestures (was 5)
COOLDOWN_PERIOD = 4.0


class QuoriGestureDetection:

    # --- Quori gestures: wave, "fist", "woohoo-esque"
    wave_positions = [[-0.5, 0.2, 0.0, 0.0, 0.0], [-0.5, 0.2, 0.5, 0.1, 0.0], [-0.5, 0.2, 0.5, 0.4, 0.0], [-0.5, 0.2, 0.5, 0.1, 0.0], [-0.5, 0.2, 0.0, 0.0, 0.0]]
    fist_pos = [1.22, 0.0, 0.0, 0.0]
    both_hands_pos = [0.5, 0.5, 0.0, 0.0]  # example for peace/ok both arms
    meaningful_gestures = {"fist", "open_palm", "thumbs_up", "peace_sign", "ok_sign", "hang_loose", "horns", "blurred"}
    
    last_published_label = "None"
    last_published_time = 0.0
    detection_counts = {}


    #   /quori/arm_<side>/cmd_pos_dir   (Vector3: x=outer motor rad, y=inner motor rad, z=time or -1)
    #   /quori/arm_<side>/shoulder_pos  (Vector3: x=J1 rad, y=J2 rad, z=unused)

    def __init__(self):
        rospy.init_node('hand_classifier_quori_node', anonymous=True)

        # Useful Publishers
        self.image_pub = rospy.Publisher("/hand_image_blurred", Image, queue_size=1)
        self.arm_pub = rospy.Publisher('/quori/joint_trajectory_controller/command', JointTrajectory, queue_size=10)

        # these exist, but for no good reason?
        #i was using these when i had a sound client publishing the msgs 
        self.hand_pub = rospy.Publisher('/quori/hand_classifier/gesture', String, queue_size=10)
        self.trans_pub = rospy.Publisher('/whisper/transcript', String, queue_size=2)
        
        # AI Hallucinations? yes
        # st = rospy.get_param('~arm_side', 'right')  # use 'right' or 'left' as appropriate
        # self.arm_mcmd_pub = rospy.Publisher(f'/quori/arm_{st}/cmd_pos_dir', Vector3, queue_size=1)
        # self.arm_joint_pub = rospy.Publisher(f'/quori/arm_{st}/shoulder_pos', Vector3, queue_size=1)

        self.classifier = HandClassifier("/home/quori6/catkin_ws/src/comic_con/src/hand_signs.pickle")
        self.bridge = CvBridge()
        self.tts_engine = pyttsx3.init()
        voices = self.tts_engine.getProperty("voices")
        self.tts_engine.setProperty("voice", voices[30].id)

        rospy.Subscriber("/astra_ros/devices/default/color/image_color", Image, self.image_callback)
        

    def move_arm(self, joint_pub, joint_angles, duration=2.0):
        """
        Publish a single-point JointTrajectory using the provided joint_pub.
        If joint_pub is None, fall back to the module-level traj_pub.
        """
        pub = joint_pub
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

    def perform_wave(self):
        # 5 joints: [l_shoulder_pitch, l_shoulder_roll, r_shoulder_pitch, r_shoulder_roll, waist_pitch]
        wave_steps = self.wave_positions
        for step in wave_steps:
            self.move_arm(self.arm_pub, step, duration=0.4)
            rospy.sleep(0.4)


    def speak(self, message):
        """Non-Blocking wrapper for the speak thread"""
        # FYI this is a cool python thing because it should throw a type error if it fully evaluated this line...
        if not hasattr(self, 'speak_thread') or not self.speak_thread.is_alive():
            self.speak_thread = threading.Thread(target=self._speak, args=(message,))
            self.speak_thread.start()

    def _speak(self, message):
        try:
            # tts = gTTS(text=message)
            # filepath = '/home/quori6/message.mp3'
            # tts.save(filepath)
            # os.system(f'mpg123 {filepath} && rm {filepath}')
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            rospy.logwarn(f"tts failed: {e}")      


    def image_callback(self, msg):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge error (bgr8): {e}")
            try:
                # If 'bgr8' fails, try 'rgb8' as a fallback.
                cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            except Exception as e:
                rospy.logerr(f"CvBridge error (rgb8): {e}")
                return 

        label, dist = self.classifier.classify(cv_image, is_bgr=True, display=True)

        rospy.logdebug(f"[hand_classifier] candidate='{label}' mean_dist={dist}")

    
        for k in list(self.detection_counts.keys()):
            if k != label:
                self.detection_counts[k] = 0
        self.detection_counts[label] = self.detection_counts.get(label, 0) + 1

        now = time.time()
        should_publish = False

        rospy.logdebug(f"[hand_classifier] DETECTION_COUNTS={self.detection_counts} label={label} dist={dist:.2f} last_published={self.last_published_label}")

        # Only publish if candidate is meaningful, confident, and stable for N frames
        if dist <= DIST_THRESHOLD and label in self.meaningful_gestures:
            if self.detection_counts.get(label, 0) >= STABLE_FRAMES_REQUIRED:
                # still apply cooldown to avoid constant triggers

                if label != self.last_published_label and (now - self.last_published_time) > COOLDOWN_PERIOD:
                    should_publish = True

        if should_publish:
            self.hand_pub.publish(label)

            self.last_published_label = label
            self.last_published_time = now
            rospy.loginfo(f"Gesture accepted: {label} (dist={dist}) after {self.detection_counts[label]} frames")
            if self.arm_pub is None:
                rospy.logwarn("arm_pub is not initialized, cannot move arms")
            else:
                responses = []
                if label == "peace_sign":
                    responses = ["Peace, dude!", "Peace Out!", "Peace and Love", "Peace", "No gang signs please...No throw it up, I love peace"]
                elif label == "fist":
                    responses = ["I'd fist bump you if I could...", "Oh, yeah!", "Wooohooo!"]
                elif label == "ok_sign":
                    responses = ["Oak E Doke E!", "Ay Okay!", "Okie Dokie, artichokie", "Perfect!"]
                elif label == "open_palm":
                    responses = ["Hey!", "Hello", "Hey There", "Hiya!"]
                elif label == "thumbs_up":
                    responses = ["Thumbs up!", "You got it!", "Awesome!", "Great job!"]
                elif label == "horns":
                    responses = ["Rock on!", "Keep it cool!", "You rule!", "Stay awesome!"]
                elif label == "hang_loose":
                    responses = ["Hang loose!", "Take it easy!", "Chill out!", "No worries!"]
                if not hasattr(self, 'wave_thread') or not self.wave_thread.is_alive():
                    self.wave_thread = threading.Thread(target=self.perform_wave)
                    self.wave_thread.start()
                if responses:
                    response = random.choice(responses)
                    self.speak(response)
        else:
            self.hand_pub.publish("None")
            rospy.logdebug("No meaningful gesture published (unstable/confident)")
    
        color = (0, 255, 0) if dist <= DIST_THRESHOLD else (0,0,255)

        cv2.putText(cv_image, f"{label} ({np.round(dist, 2)})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        detection = self.classifier.detector.detect(cv_image)
        if label == "blurred" and getattr(detection, "multi_hand_landmarks", None) is not None and detection.multi_hand_landmarks is not None:
            hand_landmarks = detection.multi_hand_landmarks[0]
            x, y, w_box, h_box = self.bbox(hand_landmarks, cv_image.shape)
            roi = cv_image[y:y+h_box, x:x+w_box]
            blurred_roi = cv2.GaussianBlur(roi, (41, 41), 0)
            cv_image[y:y+h_box, x:x+w_box] = blurred_roi

        # Display the image in a window and keep it updated.

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        cv2.imshow("Quori Hand Classifier", cv_image)
        cv2.waitKey(1)

    def bbox(self, hand_landmarks, img_shape, margin=5):
        h, w, _ = img_shape
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
        
        x_min = max(0, min(x_coords) - margin)
        x_max = min(w, max(x_coords) + margin)
        y_min = max(0, min(y_coords) - margin)
        y_max = min(h, max(y_coords) + margin)
        
        return x_min, y_min, x_max - x_min, y_max - y_min


def main():

    QuoriGestureDetection()
    rospy.loginfo("Hand classifier node started. Waiting for images...")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
