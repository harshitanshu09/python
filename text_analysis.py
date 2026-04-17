from transformers import pipeline

print("Loading models... please wait")

# Load models
sentiment_model = pipeline("sentiment-analysis")

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

print("Models loaded successfully!")


def analyze_text(text):
    sentiment = sentiment_model(text)
    emotion = emotion_model(text)

    return sentiment, emotion


if __name__ == "__main__":
    text = input("Enter text: ")

    sentiment, emotion = analyze_text(text)

    print(f"\nInput Text: {text}")
    print("\n========== RESULT ==========")

    # Sentiment
    sent = sentiment[0]
    print(f"Sentiment: {sent['label']} ({round(sent['score'], 3)})")

    # Emotion handling
    if isinstance(emotion[0], list):
        emotions_list = emotion[0]
    else:
        emotions_list = emotion

    top_emotion = max(emotions_list, key=lambda x: x["score"])

    print(
        f"Top Emotion: {top_emotion['label']} "
        f"({round(top_emotion['score'], 3)})"
    )

    print("============================")
