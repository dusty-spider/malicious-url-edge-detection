import sys
from pathlib import Path

from flask import Flask, request, jsonify

# The predictor lives in 04-Deployment/; make it importable from this folder too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04-Deployment"))
from pi_predict import predict_url


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"error": "request must contain a 'url' field"}), 400

    url = data["url"].strip()
    if not url:
        return jsonify({"error": "url cannot be empty"}), 400

    result = predict_url(url)
    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # Lab-only: no authentication here; do not expose outside a trusted network.
    app.run(host="0.0.0.0", port=5000, debug=False)
