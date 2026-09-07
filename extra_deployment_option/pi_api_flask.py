
from flask import Flask, request, jsonify
from pi_predict import predict_url
#app creation
app = Flask(__name__)


@app.route("/predict", methods=["POST"]) #only post requests obv
def predict():
    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"error": "request must contain a 'url' field"}), 400 #haha kenny didnt think i needed webapp so fast

    url = data["url"].strip()
    if not url:
        return jsonify({"error": "url cannot be empty"}), 400

    result = predict_url(url) #call to the model
    return jsonify(result), 200 #return with success

#health endpoint i cba not knowing if its up
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # host 0.0.0.0 so it's reachable from outside the Pi over the network
    app.run(host="0.0.0.0", port=5000, debug=False)