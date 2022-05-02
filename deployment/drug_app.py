import os
import pickle
from flask import Flask, jsonify, request

# create app
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    # index/home page of the app
    # return content and status code
    return "<h1>Drug Effectiveness App</h1>", 200

# /predict endpoint


@app.route("/predict", methods=["GET"])
def predict():
    drug = request.args.get("Drug", "")
    age = request.args.get("Ags", "")
    condition = request.args.get("Condition", "")
    season = request.args.get("Season", "")
    ease_of_use = request.args.get("EaseofUse", "")
    satisfaction = request.args.get("Satisfaction", "")
    sex = request.args.get("Sex", "")
    # print values for debugging
    prediction = predict_effectiveness(
        [drug, age, condition, season, ease_of_use, satisfaction, sex])
    # if anything goes wrong in predict_interviewed_well, it'll return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "error making prediction", 400


def predict_effectiveness(instance):
    # TODO: do this
    # unpickle our ML model here
    # then next is TDB. depends on the model we use
    return


if __name__ == "__main__":
    port = os.environ.get("PORT", 5001)
    # set to False when deployed
    app.run(debug=False, port=port, host="0.0.0.0")
