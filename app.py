import streamlit as st
from sentiment import predict_sentiment  # Import the sentiment analysis function

# Title of the app
st.title("Sentiment Analysis of Tweets")

# Subtitle
st.subheader("Analyze the sentiment of a tweet (Positive, Neutral, Negative)")

# Input text box for user to enter a tweet
user_input = st.text_area("Enter a tweet to analyze sentiment:")

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():  # Check if input is not empty
        # Predict sentiment
        result = predict_sentiment(user_input)
        # Display the result
        st.success(f"The sentiment of the tweet is: **{result}**")
    else:
        st.error("Please enter a valid tweet.")