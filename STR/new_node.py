#!/usr/bin/env python3
import os
import json, re
import subprocess
import tempfile
from sound_play.msg import SoundRequest

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
import time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import whisper
class PointingNode:
    def __init__(self):
        rospy.init_node("pointing_vlm_node", anonymous=True)

        #self.soundhandle = SoundClient()
        self.annotated_pub = rospy.Publisher("/annotated_image", Image, queue_size=1)
        self.model=whisper.load_model("base")
        self.text_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)

        self.classifier = HandClassifier("hand_signs.pickle")
        self.vtt = VisionToText()
        self.bridge = CvBridge()
        self.frame_queue = Queue(maxsize=2)
        self.vlm_queue = Queue(maxsize=1)

        # Only need one stable label tracker
        self.last_stable_label = ""
        # keep a slightly longer window for voting (used for diagnostics only)
        self.recent_descriptions = deque(maxlen=6)

        # Debounce / stability parameters (tweak with params)
        # Reduced defaults for faster acceptance: fewer votes + shorter persistence
        self.min_votes = int(rospy.get_param('~min_votes', 2))              # consecutive occurrences required
        self.min_stable_seconds = float(rospy.get_param('~min_stable_seconds', 0.8))
        self.vlm_rate_limit = float(rospy.get_param('~vlm_rate_limit', 0.5))
        self.last_vlm_time = rospy.Time(0)

        # candidate state for commit-after-persist
        self.candidate_label = None
        self.candidate_since = rospy.Time(0)
        self.candidate_votes = 0

        # simple blacklist / rejection heuristics (uncertain responses)
        self.reject_phrases = ["i can't", "i cannot", "can't determine", "i can't determine", "unable to", "i cannot determine"]
        # allow max words in accepted label (reject long descriptive phrases)
        self.max_label_words = int(rospy.get_param('~max_label_words', 3))

        # cooldown: don't speak more than once every 10 seconds
        self.cooldown_seconds = 5.0
        self.last_spoken_time = rospy.Time(0)
        # track what label we actually last spoke
        self.last_spoken_label = ""

        # speech queue to avoid overlapping speech
        self.speech_queue = Queue(maxsize=2)
        threading.Thread(target=self.speech_worker, daemon=True).start()

        # Worker threads
        threading.Thread(target=self.frame_worker, daemon=True).start()
        threading.Thread(target=self.vlm_worker, daemon=True).start()

        image_topic = '/astra_ros/devices/default/color/image_color'
        rospy.loginfo(f"Subscribing to image topic: {image_topic}")
        rospy.loginfo("new file")
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)

        # OpenAI setup
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def transcribe(self, audio_file):
        result = self.model.transcribe(audio_file)
        return result["text"]

    def query_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # or gpt-4
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            rospy.logwarn(f"LLM request failed: {e}")
            return "I see something, but I can't describe it right now."
    def publish_speech_marker(self, text, position = (0,0,1.5), marker_id=0):
        marker = Marker()
        marker.header.frame_id = "quori/base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "speech_bubble"
        marker.id = marker_id + 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2] + 0.2
        marker.pose.orientation.w = 1.0 # no rotation
        marker.scale.x = 0.3 # width
        marker.scale.y = 0.3 # height
        marker.scale.z = 0.5 # font size
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = text
        self.text_pub.publish(marker)
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

                # rate-limit VLM calls to reduce chattiness & flips
                now = rospy.Time.now()
                if (now - self.last_vlm_time).to_sec() < self.vlm_rate_limit:
                    self.vlm_queue.task_done()
                    continue
                description = self.vtt.viz_to_text(
                    img=crop,
                    prompt="What specific, concise object is being pointed to? Just name the object. It cannot be a human, something a human is wearing on their body like jewelry or clothes, or a body part."
                )
                self.last_vlm_time = rospy.Time.now()

                if not description:
                    self.vlm_queue.task_done()
                    continue

                # normalize and simple filtering
                cleaned = re.sub(r'[^a-z0-9\s]', '', description.lower()).strip()
                if any(phrase in cleaned for phrase in self.reject_phrases):
                    rospy.loginfo(f"VLM returned uncertain/verbose response, ignoring: '{description}'")
                    self.vlm_queue.task_done()
                    continue
                word_count = len(cleaned.split())
                if word_count == 0 or word_count > self.max_label_words:
                    rospy.loginfo(f"VLM returned too-long/empty label, ignoring: '{description}'")
                    self.vlm_queue.task_done()
                    continue

                # store for diagnostics
                self.recent_descriptions.append(cleaned)

                # candidate logic: require consecutive persistence + time
                if cleaned != self.candidate_label:
                    self.candidate_label = cleaned
                    self.candidate_since = rospy.Time.now()
                    self.candidate_votes = 1
                else:
                    self.candidate_votes += 1

                elapsed_candidate = (rospy.Time.now() - self.candidate_since).to_sec()
                if (self.candidate_votes >= self.min_votes and
                    elapsed_candidate >= self.min_stable_seconds and
                    self.candidate_label != self.last_stable_label):
                    # commit stable label
                    self.last_stable_label = self.candidate_label
                    rospy.loginfo(f"Stable label updated: {self.last_stable_label} (votes={self.candidate_votes}, time={elapsed_candidate:.2f}s)")

                    # enforce cooldown: only block if it's the same label we last spoke
                    now2 = rospy.Time.now()
                    elapsed_since_spoken = (now2 - self.last_spoken_time).to_sec()
                    if elapsed_since_spoken < self.cooldown_seconds and self.last_stable_label == self.last_spoken_label:
                        rospy.loginfo(f"Cooldown active ({elapsed_since_spoken:.1f}s) for label '{self.last_stable_label}'. Skipping speech.")
                    else:
                        prompt = f"The user is pointing at a {self.last_stable_label}. Write a fun, one-line sentence as a narrator from a comic movie about the object being pointed to."
                        speech_text = self.query_llm(prompt)
                        self.publish_speech_marker(speech_text)

                        try:
                            if self.speech_queue.full():
                                try:
                                    self.speech_queue.get_nowait()
                                    self.speech_queue.task_done()
                                except Exception:
                                    pass
                            self.speech_queue.put_nowait((self.last_stable_label, speech_text))
                        except Exception as e:
                            rospy.logwarn(f"Failed to enqueue speech: {e}")
               # reset candidate state so it must re-accumulate
                    self.candidate_votes = 0
                    self.candidate_label = None
                    self.candidate_since = rospy.Time(0)

                self.vlm_queue.task_done()
            except Empty:
                 continue

    def speech_worker(self):
        """Dequeue and play one speech at a time to avoid overlap. Updates last_spoken_* after playback."""
        while not rospy.is_shutdown():
            try:
                label, text = self.speech_queue.get(timeout=1)
            except Empty:
                continue

            # stop any existing playback to avoid overlap, then publish a SAY request
            try:
                with tempfile.NamedTemporaryFile(delete = False, suffix=".mp3") as f:
                    speech_file = f.name
                with self.client.audio.speech.with_streaming_response.create(model = "gpt-4o-mini-tts", voice="alloy", input=text,) as response:
                    response.stream_to_file(speech_file)
                subprocess.call(["mpg123", "-q", speech_file])

            except Exception as e:
                rospy.logwarn(f"TTS failed: {e}")
            self.last_spoken_time=rospy.Time.now()
            self.last_spoken_label = label

            try:
                self.speech_queue.task_done()
            except Exception:
                pass



  


    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = PointingNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS shutdown requested")
