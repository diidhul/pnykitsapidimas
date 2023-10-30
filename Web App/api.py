from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
api = Api(app)

# Load the pre-trained model and other required data
data = pd.read_excel("/home/tinapyp/code/Side Hustle/Cow Disease Detection/data_terbaru.xlsx")
data = data.dropna()
data['Gejala'] = data['Gejala'].str.replace('[^\w\s]', '')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
data['Gejala'] = data['Gejala'].apply(lambda x: stemmer.stem(x))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Gejala'])
y = data['Diagnosa']
clf = RandomForestClassifier(criterion='entropy', n_estimators=200, random_state=1)
clf.fit(X, y)

# Define a resource to handle API requests
class SymptomDiagnosis(Resource):
    def get(self):
        return {"message": "Welcome to the Cow Disease Diagnosis API."}

    def post(self):
        user_input = request.json.get("symptoms")
        user_input = [stemmer.stem(user_input)]
        user_input = vectorizer.transform(user_input)
        prediction = clf.predict(user_input)[0]

        # Find Penanganan and Resiko based on the prediction
        result = data[data['Diagnosa'] == prediction].iloc[0]
        penanganan = result['Penanganan']
        risiko = result['Risiko']

        response = {
            "diagnosis": prediction,
            "penanganan": penanganan,
            "risiko": risiko
        }
        return response

# Add the API resource to your app
api.add_resource(SymptomDiagnosis, '/diagnose')

if __name__ == '__main__':
    app.run(debug=True)
