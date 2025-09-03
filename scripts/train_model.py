#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import joblib
from responses import keyword_responses, responses

# Load your CSV
df = pd.read_csv("Small_talk_Intent.csv")

# Add extra examples for 'smalltalk_agent_acquaintance'
name_variations = [
    "What is your name?",
    "Who are you?",
    "May I know your name?",
    "Tell me your name",
    "Can you introduce yourself?",
    "Your name please?",
    "Do you have a name?",
    "How should I address you?",
    "I want to know your name",
    "Who am I talking to?",
    "Could you tell me your name?",
    "What's your name?",
    "I’d like to know your name",
    "Introduce yourself, please",
    "What do people call you?"
]

interest_examples = [
    "What's your favorite movie?",
    "Do you like any movies?",
    "What movies do you enjoy?",
    "Tell me your favorite film",
    "Any favorite movies?"
]

# Add new examples to the DataFrame
new_examples = [(utt, "smalltalk_agent_acquaintance") for utt in name_variations]
df_new = pd.DataFrame(new_examples, columns=["Utterances", "Intent"])
df = pd.concat([df, df_new], ignore_index=True)

# Add extra examples for interests
new_examples = [(utt, "smalltalk_agent_interests") for utt in interest_examples]
df_new = pd.DataFrame(new_examples, columns=["Utterances", "Intent"])
df = pd.concat([df, df_new], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42)

# Separate features (X) and labels (y)
X = df["Utterances"].tolist()
y = df["Intent"].tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences into embeddings
X_train_emb = embed_model.encode(X_train, convert_to_numpy=True)
X_test_emb = embed_model.encode(X_test, convert_to_numpy=True)

# Train a Logistic Regression classifier on the embeddings
clf = LogisticRegression(max_iter=500)
clf.fit(X_train_emb, y_train)

# Evaluate the model's accuracy on the test set
accuracy = clf.score(X_test_emb, y_test)
print(f"Model trained and evaluated.")
print(f"Accuracy: {accuracy:.4f}")

# Save the trained classifier and the embedder for later use
joblib.dump(clf, "intent_classifier.pkl")
joblib.dump(embed_model, "sentence_embedder.pkl")
print("Classifier and embedder saved as 'intent_classifier.pkl' and 'sentence_embedder.pkl'")

# Optional: Test the saved model to ensure it works correctly
print("\n--- Testing Saved Model ---")
classifier_test = joblib.load("intent_classifier.pkl")
embedder_test = joblib.load("sentence_embedder.pkl")

test_cases = [
    "What is your name?",
    "Who are you?",
    "Tell me your name",
    "Can you introduce yourself?",
]

def predict_intent(user_input):
    emb = embed_model.encode([user_input], convert_to_numpy=True)
    pred = clf.predict(emb)[0]
    return pred


for utterance in test_cases:
    embedding = embedder_test.encode([utterance])
    predicted_intent = classifier_test.predict(embedding)[0]
    result = "PASS" if predicted_intent == "smalltalk_agent_acquaintance" else "FAIL"
    print(f"Utterance: '{utterance}' | Predicted: {predicted_intent} | {result}")
