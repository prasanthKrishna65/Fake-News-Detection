from flask import Flask, request, jsonify
import fasttext

app = Flask(__name__)
model = fasttext.load_model("best_fasttext_model.bin")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    label, confidence = model.predict(text)
    label_num = int(label[0].replace("__label__", ""))
    return jsonify({"label": label_num, "confidence": round(confidence[0], 4)})

if __name__ == "__main__":
    app.run(debug=True)
