#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import joblib
import random
from responses import keyword_responses, responses
from sentence_transformers import SentenceTransformer

# Load the saved model and embedder
# This assumes 'intent_classifier.pkl' and 'sentence_embedder.pkl' were saved by train_model.py
embed_model = joblib.load("sentence_embedder.pkl")
clf = joblib.load("intent_classifier.pkl")




def predict_intent(user_input):
    """Predicts the intent of the user's input using the loaded model."""
    emb = embed_model.encode([user_input], convert_to_numpy=True)
    pred = clf.predict(emb)[0]
    return pred

def get_response(user_input):
    """
    Selects a response based on keywords or predicted intent.
    This function has been corrected to check all keywords first.
    """
    # 1. Check for keyword-based responses first
    for keyword, response in keyword_responses.items():
        if keyword.lower() in user_input.lower():
            return response

    # 2. If no keyword matches, predict the intent
    pred_intent = predict_intent(user_input)
    
    # 3. Get a random response for that intent, or a default message
    return random.choice(responses.get(pred_intent, ["I'm not sure how to respond to that."]))

def callback(input_msg):
    user_input = input_msg.data
    rospy.loginfo("User input received: %s", user_input)
    reply = get_response(user_input)
    rospy.loginfo("Responding with: %s", reply)
    # Here you would publish the 'reply' to a ROS topic

# The rest of your ROS node setup (rospy.init_node, etc.) would follow.
def main():
    """Initializes the ROS node and sets up the subscriber and publisher."""
    global pub
    
    # 1. Initialize the ROS node
    rospy.init_node('comic_con_chatbot', anonymous=True)
    
    # 2. Create a subscriber to listen for user input
    rospy.Subscriber('user_input', String, callback)
    
    # 3. Create a publisher to send robot replies
    pub = rospy.Publisher('robot_reply', String, queue_size=10)
    
    # 4. Wait for messages. This keeps the node from exiting.
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
