#!/usr/bin/env python3
import os
import json, re
from collections import Counter, deque
import openai
from openai import OpenAI
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from classify_hand import HandClassifier
from vlm_basic import VisionToText
import threading

from queue import Queue, Empty
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient

class PointingNode:
    def __init__(self):
        rospy.init_node("pointing_vlm_node", anonymous=True)

        self.speech_pub = rospy.Publisher("/robotsound", SoundRequest, queue_size=10)
        self.annotated_pub = rospy.Publisher("/annotated_image", Image, queue_size=1)

        self.classifier = HandClassifier("hand_signs.pickle")
        self.vtt = VisionToText()
        self.bridge = CvBridge()
        #self.soundhandle = SoundClient()

        self.frame_queue = Queue(maxsize=2)
        self.vlm_queue = Queue(maxsize=1)

        # Only need one stable label tracker
        self.last_stable_label = ""
        self.recent_descriptions = deque(maxlen=5)

        # Worker threads
        threading.Thread(target=self.frame_worker, daemon=True).start()
        threading.Thread(target=self.vlm_worker, daemon=True).start()

        image_topic = '/astra_ros/devices/default/color/image_color'
        rospy.loginfo(f"Subscribing to image topic: {image_topic}")
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)

        # OpenAI setup
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def query_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # or gpt-4
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            rospy.logwarn(f"LLM request failed: {e}")
            return "I see something, but I can't describe it right now."

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        except Exception as e:
            rospy.logerr(f"CvBridge error: {e}")

    def frame_worker(self):
        while not rospy.is_shutdown():
            try:
                frame = self.frame_queue.get(timeout=1)
                self.detect_and_crop(frame)
                self.frame_queue.task_done()
            except Empty:
                continue

    def detect_and_crop(self, frame):
        label, dist = self.classifier.classify(frame)
        detected_hands = self.classifier.detector.detect(frame).multi_hand_landmarks

        # Draw last known bounding box and stable label always
        if hasattr(self, "latest_bbox") and self.latest_bbox:
            x1, y1, x2, y2 = self.latest_bbox
            if self.last_stable_label:
                cv2.putText(frame, self.last_stable_label[:30], (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if detected_hands and label == "point" :

            hands = detected_hands[0]
            wrist = hands.landmark[0]
            index = hands.landmark[8]

            h, w, _ = frame.shape
            wrist_px = (int(wrist.x * w), int(wrist.y * h))
            index_px = (int(index.x * w), int(index.y * h))
            cv2.line(frame, wrist_px, index_px, (0, 255, 0), 2)

            # Extend pointing vector
            scale = 0.5
            end_px = (int(index_px[0] + (index_px[0] - wrist_px[0]) * scale),
                      int(index_px[1] + (index_px[1] - wrist_px[1]) * scale))

            crop_size = 200
            x1 = max(0, end_px[0] - crop_size // 2)
            y1 = max(0, end_px[1] - crop_size // 2)
            x2 = min(w, end_px[0] + crop_size // 2)
            y2 = min(h, end_px[1] + crop_size // 2)

            # Save bbox
            self.latest_bbox = (x1, y1, x2, y2)

            # --- Stability check ---
            center = ((x1 + x2)//2, (y1 + y2)//2)
            if not hasattr(self, "point_history"):
                self.point_history = deque(maxlen=5)
            self.point_history.append(center)

            # Compute average movement
            if len(self.point_history) > 1:
                movement = np.mean([np.linalg.norm(np.array(self.point_history[i]) - np.array(self.point_history[i-1]))
                                    for i in range(1, len(self.point_history))])
            else:
                movement = float('inf')  # treat as unstable

            # Only send to VLM if movement is small (hand pointing is stable)
            if movement < 5 and x2 > x1 and y2 > y1:
                pointed_crop = frame[y1:y2, x1:x2]
                if pointed_crop.size > 0 and not self.vlm_queue.full():
                    self.vlm_queue.put(pointed_crop)

        cv2.imshow("Pointing Detection", frame)
        cv2.waitKey(1)

    def vlm_worker(self):
        while not rospy.is_shutdown():
            try:
                crop = self.vlm_queue.get(timeout=1)
                description = self.vtt.viz_to_text(
                    img=crop,
                    prompt="What specific, concise object is being pointed to? Just the object. It cannot be a human or a body part."
                )
                if description:
                    self.recent_descriptions.append(description)

                    # pick the most frequent label
                    most_common = Counter(self.recent_descriptions).most_common(1)[0][0]

                    # update if changed
                    if most_common != self.last_stable_label:
                        self.last_stable_label = most_common
                        rospy.loginfo(f"Stable label updated: {self.last_stable_label}")

                        # what quori says
                        prompt = f"The user is pointing at a {self.last_stable_label}. Write a fun, narrator from a comic movie-like sentence about the object being pointed to."
                        speech_text = self.query_llm(prompt)
                        #self.soundhandle.say(speech_text)

                        msg = SoundRequest()
                        msg.sound = SoundRequest.SAY        # TTS mode
                        msg.command = SoundRequest.PLAY_ONCE
                        msg.volume = 0.8
                        msg.arg = speech_text               # the text to speak
                        self.speech_pub.publish(msg)

                self.vlm_queue.task_done()
            except Empty:
                continue

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = PointingNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS shutdown requested")
