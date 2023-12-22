from flask import Flask, render_template, jsonify, request
import script

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_api', methods=['POST'])
def run_api():
    data = script.rapidapi()
    extracted_reviews, total_number_of_reviews = script.extract_reviews(data)
    return jsonify({'total_reviews': total_number_of_reviews})


@app.route('/run_predictions', methods=['POST'])
def run_predictions():
    data = script.rapidapi()
    extracted_reviews, _ = script.extract_reviews(data)
    prediction_results = script.main(extracted_reviews)  # Use the main function to get predictions
    return jsonify(prediction_results)


if __name__ == '__main__':
    app.run(debug=True)
