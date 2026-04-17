import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_models():
    # Sentiment model
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Emotion model
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

    return sentiment_model, emotion_model


# Load models once
sentiment_model, emotion_model = load_models()


# UI
st.title("AI Sentiment & Emotion Analyzer")

text = st.text_area("Enter your text here:")


if st.button("Analyze"):

    if text.strip():

        # Run sentiment model
        sentiment = sentiment_model(text)
        sent = sentiment[0]

        # Run emotion model
        emotion = emotion_model(text)

        # Safe handling for emotion output
        if isinstance(emotion[0], list):
            emotions_list = emotion[0]
        else:
            emotions_list = emotion

        top_emotion = max(
            emotions_list,
            key=lambda x: x["score"]
        )

        # Output
        st.subheader("Analysis Result")

        st.success(
            f"Sentiment: {sent['label']}"
        )

        st.write(
            f"Confidence Score: {round(sent['score'], 2)}"
        )

        st.info(
            f"Detected Emotion: "
            f"{top_emotion['label']} "
            f"({round(top_emotion['score'], 2)})"
        )

        # Interpretation
        if sent["label"] == "POSITIVE":
            st.write(
                "Interpretation: The text expresses a positive attitude."
            )
        else:
            st.write(
                "Interpretation: The text reflects a negative sentiment."
            )

    else:
        st.warning("Please enter some text first.")
