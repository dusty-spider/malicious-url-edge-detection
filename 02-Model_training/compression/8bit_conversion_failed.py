import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report

# load model
model = tf.keras.models.load_model("models/baseline_cnn.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

X_train = np.load("X_train1.npy")

# dataset
def representative_dataset():
    for i in range(1000):
        yield [X_train[i:i+1].astype("float32")]

converter.representative_dataset = representative_dataset

# full INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# save
tflite_path = "models/cnn_8bit.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved!!!!!")

# size comparison
keras_size = os.path.getsize("models/baseline_cnn.h5") / 1024
tflite_size = os.path.getsize(tflite_path) / 1024

print(f"Keras model size: {keras_size:.2f} KB")
print(f"TFLite model size: {tflite_size:.2f} KB")


# evaluation
X_test = np.load("X_test1.npy")
y_test = np.load("y_test1.npy")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero = input_details[0]["quantization"]
output_scale, output_zero = output_details[0]["quantization"]

preds = []

for i in range(len(X_test)):
    sample = X_test[i:i+1].astype("float32")

    # quantise input
    sample_q = sample / input_scale + input_zero
    sample_q = sample_q.astype("int8")

    interpreter.set_tensor(input_details[0]['index'], sample_q)
    interpreter.invoke()

    output_q = interpreter.get_tensor(output_details[0]['index'])

    # dequantise output
    output = (output_q - output_zero) * output_scale
    preds.append(output[0][0])

y_pred = (np.array(preds) > 0.5).astype(int)

print(classification_report(y_test, y_pred))

