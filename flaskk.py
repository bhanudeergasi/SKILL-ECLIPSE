from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import logging
import pickle
import base64
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load NLTK stopwords
STOPWORDS = set(stopwords.words("english"))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set a maximum request size (e.g., 16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.before_request
def log_request():
    logging.debug(f"Request received: {request.method} {request.path}")
    logging.debug(f"Request data: {request.data}")

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Load models
    try:
        predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
        scaler = pickle.load(open("Models/scaler.pkl", "rb"))
        cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return jsonify({"error": "Error loading models"}), 500

    try:
        # Check for CSV file or JSON text input
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(predictor, scaler, cv, data)
            if predictions is None:
                return jsonify({"error": "Bulk prediction failed"}), 500

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getvalue()).decode("ascii")
            return response

        elif request.json and "text" in request.json:
            # Single string prediction
            text_input = request.json.get("text", "")
            if not text_input:
                return jsonify({"error": "No text input provided"}), 400
            
            logging.debug(f"Received text input: {text_input}")
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            if predicted_sentiment == "Error in prediction":
                return jsonify({"error": "Prediction failed"}), 500
            
            logging.debug(f"Predicted sentiment: {predicted_sentiment}")
            return jsonify({"prediction": predicted_sentiment})

        else:
            return jsonify({"error": "No valid input provided"}), 400

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

def single_prediction(predictor, scaler, cv, text_input):
    try:
        corpus = []
        stemmer = PorterStemmer()
        review = re.sub("[^a-zA-Z]", " ", text_input)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)
        
        # Transform the input text
        X_prediction = cv.transform(corpus).toarray()
        logging.debug(f"Vectorized input: {X_prediction}")
        
        # Scale the transformed data
        X_prediction_scl = scaler.transform(X_prediction)
        logging.debug(f"Scaled input: {X_prediction_scl}")
        
        # Predict the sentiment using predict_proba
        y_predictions_proba = predictor.predict_proba(X_prediction_scl)
        logging.debug(f"Predicted probabilities: {y_predictions_proba}")
        
        # Get the class with the highest probability
        y_predictions = y_predictions_proba.argmax(axis=1)[0]
        logging.debug(f"Predicted class: {y_predictions}")
        
        # Alternatively, use predict method directly (for testing)
        y_predictions_direct = predictor.predict(X_prediction_scl)[0]
        logging.debug(f"Direct prediction: {y_predictions_direct}")
        
        return "Positive" if y_predictions == 1 else "Negative"
    except Exception as e:
        logging.error(f"Error in single prediction: {e}")
        return "Error in prediction"


def bulk_prediction(predictor, scaler, cv, data):
    try:
        corpus = []
        stemmer = PorterStemmer()
        for i in range(data.shape[0]):
            review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
            review = review.lower().split()
            review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
            review = " ".join(review)
            corpus.append(review)
        X_prediction = cv.transform(corpus).toarray()
        X_prediction_scl = scaler.transform(X_prediction)
        y_predictions = predictor.predict_proba(X_prediction_scl)
        y_predictions = y_predictions.argmax(axis=1)
        y_predictions = list(map(sentiment_mapping, y_predictions))
        data["Predicted sentiment"] = y_predictions
        predictions_csv = BytesIO()
        data.to_csv(predictions_csv, index=False)
        predictions_csv.seek(0)
        graph = get_distribution_graph(data)
        return predictions_csv, graph
    except Exception as e:
        logging.error(f"Error in bulk prediction: {e}")
        return None, None

def get_distribution_graph(data):
    try:
        fig = plt.figure(figsize=(5, 5))
        colors = ("green", "red")
        wp = {"linewidth": 1, "edgecolor": "black"}
        tags = data["Predicted sentiment"].value_counts()
        explode = (0.01, 0.01)
        tags.plot(
            kind="pie",
            autopct="%1.1f%%",
            shadow=True,
            colors=colors,
            startangle=90,
            wedgeprops=wp,
            explode=explode,
            title="Sentiment Distribution",
            xlabel="",
            ylabel="",
        )
        graph = BytesIO()
        plt.savefig(graph, format="png")
        plt.close()
        return graph
    except Exception as e:
        logging.error(f"Error in generating graph: {e}")
        return None

def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=False)

