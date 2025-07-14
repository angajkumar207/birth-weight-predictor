from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Function to clean and format input data
def get_cleaned_data(form_data):
    try:

        Birth = int(form_data['Birth.Weight'])
        gestation = int(form_data['Gestational.Days'])
        age = int(form_data['Maternal.Age'])
        Height = int(form_data['Maternal.Height'])
        weight = int(form_data['Maternal.Pregnancy.Weight'])

        cleaned_data = {
            "Birth.Weight":[Birth],
            "Gestational.Days": [gestation],
            "Maternal.Age": [age],
            "Maternal.Height": [Height],
            "Maternal.Pregnancy.Weight": [weight]
        }

        return cleaned_data
    except KeyError as e:
        raise ValueError(f"Missing field: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=['POST'])
def get_prediction():
    try:
        # Get form data from HTML
        baby_data_form = request.form

        # Clean the data
        baby_data_cleaned = get_cleaned_data(baby_data_form)

        # Convert to DataFrame
        baby_df = pd.DataFrame(baby_data_cleaned)

        # Load pre-trained model
        model_path = os.path.join("model", "model.pkl")
        with open(model_path, 'rb') as obj:
            model = pickle.load(obj)

        # Make prediction
        prediction = model.predict(baby_df)
        prediction = round(float(prediction), 2)

        # Render result in template
        return render_template("index.html", prediction=prediction)

    except Exception as e:
        # Show error in browser
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
