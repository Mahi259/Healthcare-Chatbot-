from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statistics import mode

app = Flask(__name__)

# Load data
data = pd.read_csv("C:/Users/91639/Desktop/mahi's clg/chitti bot/Original_Dataset.csv")

# Combine symptom columns into a single string
data['Symptoms'] = data[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 
                         'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 
                         'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
                         'Symptom_16', 'Symptom_17']].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)

# Encode target labels
encoder = LabelEncoder()
data['Disease'] = encoder.fit_transform(data['Disease'])

# Prepare features and labels
X = data['Symptoms']
y = data['Disease']

# Vectorize symptoms
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train models
final_svm_model = SVC(probability=True)
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier()

final_svm_model.fit(X_train, y_train)
final_nb_model.fit(X_train.toarray(), y_train)
final_rf_model.fit(X_train, y_train)

# Doctor recommendations data
doctor_data = pd.read_csv("C:/Users/91639/Desktop/mahi's clg/chitti bot/Doctor_Versus_Disease.csv")

def recommendDoctors(disease):
    recommendations = []
    for _, row in doctor_data.iterrows():
        if row['Disease'] == disease:
            recommendations.append({
                "disease": row['Disease'],
                "specialist": row['Specialist'],
                "doctor": row['Doctor'],
                "contact": row['Contact']
            })
    # Ensure unique recommendations
    unique_recommendations = {rec['doctor']: rec for rec in recommendations}.values()
    return list(unique_recommendations)

def predictDisease(symptoms):
    input_data = vectorizer.transform([symptoms])
    
    rf_prediction_index = final_rf_model.predict(input_data)[0]
    nb_prediction_index = final_nb_model.predict(input_data.toarray())[0]
    svm_prediction_index = final_svm_model.predict(input_data)[0]

    rf_prediction = encoder.inverse_transform([rf_prediction_index])[0]
    nb_prediction = encoder.inverse_transform([nb_prediction_index])[0]
    svm_prediction = encoder.inverse_transform([svm_prediction_index])[0]

    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
    
    return {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    predictions = predictDisease(symptoms)
    recommendations = recommendDoctors(predictions['final_prediction'])
    return render_template('result.html', symptoms=symptoms, predictions=predictions, recommendations=recommendations)

@app.route('/evaluate', methods=['GET'])
def evaluate():
    # Make predictions on the test set
    rf_predictions = final_rf_model.predict(X_test)
    nb_predictions = final_nb_model.predict(X_test.toarray())
    svm_predictions = final_svm_model.predict(X_test)

    # Calculate accuracy
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    return f"""
        <h1>Model Accuracy</h1>
        <p>Random Forest Accuracy: {rf_accuracy * 100:.2f}%</p>
        <p>Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%</p>
        <p>SVM Accuracy: {svm_accuracy * 100:.2f}%</p>
    """

if __name__ == '__main__':
    app.run(debug=True)
