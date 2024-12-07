import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

IMG_HEIGHT = 96
IMG_WIDTH = 96

# Load the saved Keras model
model = load_model("model_01.keras")

# Define the labels for ASL classes
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y']  

def preprocess_frame(frame):
    """Preprocess the video frame for the ASL model."""
    # Convert the frame to a TensorFlow tensor
    if isinstance(frame, np.ndarray):
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        # Reshape to add channel dimension if grayscale
        if frame.ndim == 2:  # If the input is grayscale
            frame = tf.expand_dims(frame, axis=-1)
            frame = tf.image.grayscale_to_rgb(frame)

    # Ensure the frame has 3 channels (RGB)
    if frame.shape[-1] == 1:  # Grayscale image
        frame = tf.image.grayscale_to_rgb(frame)

    # First scale down to dataset dimensions (if applicable)
    frame = tf.image.resize(frame, [28, 28])  # Resize to smaller dimensions for consistency

    # Resize to the target model input dimensions
    frame = tf.image.resize(frame, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize pixel values to [0, 1]
    frame = tf.cast(frame, tf.float32) / 255.0

    # Add batch dimension for model input
    frame = tf.expand_dims(frame, axis=0)

    return frame

def preprocess_frame_cnn(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(cv2.resize(img, (64, 64)), axis = 0)
    return img


def predict_asl(frame):
    """Predict the ASL sign and return the label and probabilities."""
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)  # Predict probabilities
    predicted_label = labels[np.argmax(predictions)]  # Get the class with the highest probability

    # Generate a bar chart for probabilities
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, predictions[0])
    ax.set_title("Class Probabilities")
    ax.set_ylabel("Probability")
    ax.set_xlabel("ASL Classes")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    plt.tight_layout()

    return predicted_label, fig

css = """.my-group {max-width: 500px !important; max-height: 500px !important;}
            .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        gr.Markdown("# ASL Recognition App")
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam Input")
        with gr.Column(scale=1):
            output_label = gr.Label(label="Predicted ASL Sign")
            output_plot = gr.Plot(label="Class Probabilities")

    def gradio_pipeline(frame):
        predicted_label, fig = predict_asl(frame)
        return predicted_label, fig

    input_img.stream(gradio_pipeline, [input_img], [output_label, output_plot], time_limit=300, stream_every=0.5)

demo.launch()
