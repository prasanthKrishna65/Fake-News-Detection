from flask import Flask, request, jsonify
import fasttext
import os
import gdown

# Google Drive file ID of your model
MODEL_PATH = "best_fasttext_model.bin"
DRIVE_FILE_ID = "1LZxdIv1O6SA1Pagb0j6Q59rbI8JI3IHs"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete.")

# Download and load model
download_model()
model = fasttext.load_model(MODEL_PATH)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    label, confidence = model.predict(text)
    label_num = int(label[0].replace("__label__", ""))
    return jsonify({"label": label_num, "confidence": round(confidence[0], 4)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
