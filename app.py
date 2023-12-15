import joblib
import numpy as np
import textstat
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, gunning_fog
from textblob import TextBlob
import datetime
import warnings
from flask import Flask, render_template, request
import datetime
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
import textstat

sia = SentimentIntensityAnalyzer()
warnings.filterwarnings('ignore', category=UserWarning)
rf_model = joblib.load('F:/6893-bigdata/project/random_forest_model.joblib')
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        score = int(request.form['score'])
        summary = request.form['summary']
        text = request.form['text']

        # Calculations
        text_length = minmax_scale_input(len(text),19,25738)
        summary_length = minmax_scale_input(len(summary),0,128)
        
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
        
        
        flesch_reading_ease = minmax_scale_input(textstat.flesch_reading_ease(text),-257.73,119.19)
        gunning_fog = minmax_scale_input(textstat.gunning_fog(text),1.2,138.39)
        avg_word_length_summary = minmax_scale_input(average_word_length(summary),0,114)
        avg_word_length_text = minmax_scale_input(average_word_length(text),2,26)
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

          # Make a prediction
        X = np.array(features).reshape(1, -1)
        predictions = rf_model.predict(X)

        # Pass the result to the template
        result = 'This review might not be helpful...' if predictions == 0 else 'This review is likely to be helpful!'
        return render_template('index.html', result=result)

    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)