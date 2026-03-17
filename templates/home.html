from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

crop_model = pickle.load(open("crop_model.pkl","rb"))
yield_model = pickle.load(open("yield_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/crop")
def crop():
    return render_template("crop.html")

@app.route("/yield")
def yield_page():
    return render_template("yield.html")

@app.route("/predict_crop", methods=["POST"])
def predict_crop():

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    features = [[N,P,K,temperature,humidity,ph,rainfall]]

    result = crop_model.predict(features)

    return render_template("crop.html", prediction=result[0])


@app.route("/predict_yield", methods=["POST"])
def predict_yield():

    area = float(request.form["area"])
    fertilizer = float(request.form["fertilizer"])
    pesticide = float(request.form["pesticide"])
    water = float(request.form["water"])

    features = [[area,fertilizer,pesticide,water]]

    features_scaled = scaler.transform(features)

    result = yield_model.predict(features_scaled)

    return render_template("yield.html", prediction=round(result[0],2))

if __name__ == "__main__":
    app.run(debug=True)