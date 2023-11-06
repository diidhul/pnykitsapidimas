from flask import Flask, request, render_template
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the pre-trained model and other required data
data = pd.read_excel("dataTrain.xlsx")
data = data.dropna()
data['Gejala'] = data['Gejala'].str.replace('[^\w\s]', '')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
data['Gejala'] = data['Gejala'].apply(lambda x: stemmer.stem(x))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Gejala'])
y = data['Diagnosa']
clf = RandomForestClassifier(random_state=1)
clf.fit(X, y)

# Define routes 
@app.route('/') # ini masok ke directory pertama kali dibuka
def home():
    return render_template('login.html')

@app.route('/main') # ini masok ke directory pertama kali dibuka
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST']) #'/predict' 
def predict():
    user_input = request.form['symptoms']
    user_input = [stemmer.stem(user_input)]
    user_input = vectorizer.transform(user_input)
    prediction = clf.predict(user_input)[0]

    # Find Penanganan and Resiko based on the prediction
    result = data[data['Diagnosa'] == prediction].iloc[0]
    penanganan = result['Penanganan']
    resiko = result['Risiko']
    

    return render_template('result.html', diagnosis=prediction, penanganan=penanganan, resiko=resiko)

if __name__ == '__main__':
    app.run(debug=True)
