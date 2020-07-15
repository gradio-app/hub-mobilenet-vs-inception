import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from urllib.request import urlretrieve

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

mobile_net = tf.keras.applications.MobileNetV2()
inception_net = tf.keras.applications.InceptionV3()


def classify_image_with_mobile_net(im):
    arr = im.reshape((-1, 224, 224, 3))
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)
    prediction = mobile_net.predict(arr).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}



def classify_image_with_inception_net(im):
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    im = im.resize((299, 299))
    arr = np.array(im).reshape((-1, 299, 299, 3))
    arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    prediction = inception_net.predict(arr).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}


imagein = gr.inputs.Image()
label = gr.outputs.Label(num_top_classes=3)
sample_images = [
                 ["monkey.jpg"],
                 ["rdj.jpg"],
                 ["sailboat.jpg"],
                 ["bicycle.jpg"]
]
gr.Interface(
    [classify_image_with_mobile_net, classify_image_with_inception_net],
    imagein,
    label,
    capture_session=True,
    title="MobileNet vs. InceptionNet",
    description="Compare 2 state-of-the-art machine learning models "
                "that classify images into one of 1,000 categories: MobileNet (top), "
                "a lightweight model that has an accuracy of 0.704, vs. InceptionNet"
                "(bottom), a heavier model that has an accuracy of 0.779.",
    thumbnail="thumbnail.png",
    examples=sample_images).launch();
