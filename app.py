from flask import Flask, render_template
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tcn import TCN
from tcn.tcn import ResidualBlock

# WAJIB ADA INI
app = Flask(__name__)

model = load_model(
    "model.keras",
    custom_objects={
        'TCN': TCN,
        'ResidualBlock': ResidualBlock
    }
)

scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    last_sequence = np.load("last_sequence.npy")

    pred = model.predict(last_sequence)
    pred_real = scaler.inverse_transform(pred)

    hasil = round(pred_real[0][0] * 1000)

    return render_template("index.html", prediction=hasil)

if __name__ == "__main__":
    app.run(debug=True)