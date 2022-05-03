import os
import pickle
from flask import Flask, jsonify, request

from mysklearn import myutils

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
    age = request.args.get("Age", "")
    condition = request.args.get("Condition", "")
    season = request.args.get("Season", "")
    ease_of_use = request.args.get("EaseofUse", "")
    satisfaction = request.args.get("Satisfaction", "")
    sex = request.args.get("Sex", "")
    print(drug, age, condition, season, ease_of_use, satisfaction, sex)
    # print values for debugging
    prediction = predict_effectiveness(
        [drug, age, condition, season, ease_of_use, satisfaction, sex])
    # prediction = predict_effectiveness( [age, season, ease_of_use, satisfaction, sex])
    # if anything goes wrong in predict_interviewed_well, it'll return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "error making prediction", 400


def predict_effectiveness(instance):
    # TODO: do this
    # unpickle our ML model here
    # then next is TDB. depends on the model we use
    infile = open("drug.p", "rb")
    header, trees = pickle.load(infile)
    print("header:", header)
    for tree in trees:
        print("tree", tree)
    print("TREE LENGTH:", len(trees))
    try:
        predictions = []
        for tree in trees:
            predictions.append(tdidt_predict(header, tree, instance))
            print(predictions)
        # find most common value for prediction
        # return myutils.get_most_frequent(predictions)
        return myutils.get_most_frequent(predictions)
    except:
        print("ERROR")
        return None


def tdidt_predict(header, tree, instance):
    # recursively traverse the tree
    # we need to know where we are in the tree...
    # are we at a leaf node? (base case) or attribute node
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]
    # we need to match the attribute's value in the
    # instance with the appropriate value list in the tree
    # a for loop that traverses thru each value list
    # recurse on match with instance's value
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        # grab ref to current value list
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            # we have a match, recurse
            return tdidt_predict(header, value_list[2], instance)


if __name__ == "__main__":
    port = os.environ.get("PORT", 5001)
    # set to False when deployed
    app.run(debug=False, port=port, host="0.0.0.0")
