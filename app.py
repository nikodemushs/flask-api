from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

with open('80-20/best_nb_tfidf_model_80_20.pkl', 'rb') as f:
    best_nb_cv_model = pickle.load(f)

model = best_nb_cv_model['model']
vectorizer = best_nb_cv_model['vectorizer']

label_mapping = {
    0: "Normal",
    1: "Fraud / Penipuan",
    2: "Promo"
}

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Vectorize input text
        X = vectorizer.transform([text])

        # Predict class
        pred_class_num = int(model.predict(X)[0])
        pred_class_name = label_mapping.get(pred_class_num, "Unknown")
        
        # Get probabilities for each class
        probabilities = model.predict_proba(X)[0]
        
        # Create probability dictionary
        prob_per_class = {
            f"{label_mapping.get(i, 'Unknown')}": f"{prob*100:.2f}%"
            for i, prob in enumerate(probabilities)
        }
        
        # Prepare response
        response = {
            'Prediction': pred_class_name,
            'Label': pred_class_num,
            'Probabilities':prob_per_class
        }
        
        return jsonify(response)

    except Exception as e:
        # In case of error return error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
