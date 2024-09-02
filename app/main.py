from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Class for Model Inference
class ModelInference:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, data):
        features = np.array(data).reshape(1, -1)
        prediction = self.model.predict(features)
        if(prediction[0]==0):
            return "Not Survived"
        else:
            return "Survived"

# Initialize the model
model_inference = ModelInference('app/models/model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract form data
        pclass = int(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        sex = int(request.form['sex'])
        embarked = int(request.form['embarked'])

        # Make Prediction
        prediction = model_inference.predict([pclass, age, sibsp, parch, fare, sex, embarked])

        # Redirect to result page with prediction
        return redirect(url_for('result', prediction=prediction))

    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
