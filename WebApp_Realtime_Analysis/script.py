import joblib
import requests
import numpy as np
import textstat
import nltk

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, gunning_fog
from textblob import TextBlob
import datetime
import warnings
import datetime
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat

sia = SentimentIntensityAnalyzer()
warnings.filterwarnings('ignore', category=UserWarning)
rf_model = joblib.load('./model/random_forest_model.joblib')


def main(extracted_reviews):
    # Initialize counters and lists for storing reviews
    count_0_predictions = 0
    count_1_predictions = 0
    reviews_0_predictions = []
    reviews_1_predictions = []

    # Applying featurex and model prediction to each review
    for review in extracted_reviews:
        score = review['score']
        summary = review['summary']
        text = review['text']
        X = featurex(score, summary, text)
        prediction = rf_model.predict(X)

        # Count and store the review based on the prediction
        if prediction == 0:
            count_0_predictions += 1
            reviews_0_predictions.append(review)
        else:
            count_1_predictions += 1
            reviews_1_predictions.append(review)

    # Store results in a dictionary
    result = {
        'total_0_predictions': count_0_predictions,
        'reviews_0_predictions': reviews_0_predictions,
        'total_1_predictions': count_1_predictions,
        'reviews_1_predictions': reviews_1_predictions
    }

    return result



def minmax_scale_input(input_data, min_val, max_val):
    return (input_data - min_val) / (max_val - min_val)


def average_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)


def calculate_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def type_token_ratio(text):
    tokens = text.split()
    types = set(tokens)
    if len(tokens) == 0:
        return 0  # Avoid division by zero
    return len(types) / len(tokens)


def is_weekday():
    today = datetime.datetime.now()
    return 1 if 0 <= today.weekday() < 5 else 0


def switch_input(input_value):
    return 1 if input_value == 0 else 0


def VaderPrediction(text):
    return sia.polarity_scores(text)


def rapidapi():
    url = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"
    querystring = {"asin": "B07ZPKN6YR", "country": "US", "verified_purchases_only": "false",
                   "images_or_videos_only": "false", "page": "1"}
    headers = {
        "X-RapidAPI-Key": "3ba1df3354mshbbd79a2e038ad74p1068d4jsn4c4dfcd492fe",
        "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    response = response.json()
    return response
def extract_reviews(data):
    reviews = data['data']['reviews']
    extracted_data = [{
        'score': review['review_star_rating'],
        'summary': review['review_title'],
        'text': review['review_comment']
    } for review in reviews]

    total_num = len(reviews)

    return extracted_data, total_num
def featurex(score, summary, text):
    # Calculations
    text_length = minmax_scale_input(len(text), 19, 25738)
    summary_length = minmax_scale_input(len(summary), 0, 128)

    # You can fill in the functions to calculate these values
    Vader_text = VaderPrediction(text)
    Vader_sum = VaderPrediction(summary)
    text_neg = Vader_text['neg']
    text_neu = Vader_text['neu']
    text_pos = Vader_text['pos']
    text_compound_score = Vader_text['compound']
    sum_neg = Vader_sum['neg']
    sum_neu = Vader_sum['neu']
    sum_pos = Vader_sum['pos']
    sum_compound_score = Vader_sum['compound']

    flesch_reading_ease = minmax_scale_input(textstat.flesch_reading_ease(text), -257.73, 119.19)
    gunning_fog = minmax_scale_input(textstat.gunning_fog(text), 1.2, 138.39)
    avg_word_length_summary = minmax_scale_input(average_word_length(summary), 0, 114)
    avg_word_length_text = minmax_scale_input(average_word_length(text), 2, 26)
    subjectivity_summary = calculate_subjectivity(summary)
    subjectivity_text = calculate_subjectivity(text)
    ttr_text = type_token_ratio(text)
    if_weekday_1 = is_weekday()
    if_weekday_0 = switch_input(if_weekday_1)
    features = [
        score, text_length, summary_length, text_neg, text_neu, text_pos,
        text_compound_score, sum_neg, sum_neu, sum_pos, sum_compound_score,
        flesch_reading_ease, gunning_fog, avg_word_length_summary,
        avg_word_length_text, subjectivity_summary, subjectivity_text,
        ttr_text, if_weekday_0, if_weekday_1
    ]

    X = np.array(features).reshape(1, -1)

    return X
