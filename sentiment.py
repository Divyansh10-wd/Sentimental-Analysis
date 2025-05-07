# Sentiment Analysis of Airline Tweets using Naive Bayes Classifier

# Import necessary libraries
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the dataset
sentiment_data = pd.read_csv('Tweets.csv')

# Filter rows with low confidence
sentiment_df = sentiment_data.drop(sentiment_data[sentiment_data['airline_sentiment_confidence'] < 0.5].index, axis=0)

# Define features and target
X = sentiment_df['text']
Y = sentiment_df['airline_sentiment']

# Initialize stopwords, punctuation, and lemmatizer
stop_words = stopwords.words('english')
punctuations = string.punctuation
lemmatizer = WordNetLemmatizer()

# Clean the text data
clean_data = []
for i in range(len(X)):
    text = re.sub('[^a-zA-Z]', ' ', X.iloc[i])  # Remove non-alphabetic characters
    text = text.lower().split()  # Convert to lowercase and split into words
    text = [lemmatizer.lemmatize(word) for word in text if (word not in stop_words) and (word not in punctuations)]
    text = ' '.join(text)  # Join the words back into a single string
    clean_data.append(text)

# Encode target variable
sentiments = ['negative', 'neutral', 'positive']
Y = Y.apply(lambda x: sentiments.index(x))

# Vectorize the text data
count_vectorizer = CountVectorizer(max_features=5000, stop_words=['virginamerica', 'united'])
X_fit = count_vectorizer.fit_transform(clean_data).toarray()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_fit, Y, test_size=0.3)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, Y_train)

# Function to predict sentiment for user input
def predict_sentiment(user_input):
    # Clean the user input
    text = re.sub('[^a-zA-Z]', ' ', user_input)  # Remove non-alphabetic characters
    text = text.lower().split()  # Convert to lowercase and split into words
    text = [lemmatizer.lemmatize(word) for word in text if (word not in stop_words) and (word not in punctuations)]
    text = ' '.join(text)  # Join the words back into a single string

    # Vectorize the cleaned input
    input_vector = count_vectorizer.transform([text]).toarray()

    # Predict the sentiment
    prediction = model.predict(input_vector)[0]
    return sentiments[prediction]

