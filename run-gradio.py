from imagenetlabels import idx_to_labels
import tensorflow as tf
import gradio as gr
import numpy as np


mobile_net = tf.keras.applications.MobileNetV2()
inception_net = tf.keras.applications.InceptionV3()


def classify_image_with_mobile_net(im):
	im = im.convert('RGB')
	im = im.resize((224, 224))
	arr = np.array(im).reshape((-1, 224, 224, 3))	
	arr = tf.keras.applications.mobilenet.preprocess_input(arr)
	prediction = mobile_net.predict(arr).flatten()
	return {idx_to_labels[i].split(',')[0]: float(prediction[i]) for i in range(1000)}


def classify_image_with_inception_net(im):
	im = im.convert('RGB')
	im = im.resize((299, 299))
	arr = np.array(im).reshape((-1, 299, 299, 3))
	arr = tf.keras.applications.inception_v3.preprocess_input(arr)
	prediction = inception_net.predict(arr).flatten()
	return {idx_to_labels[i].split(',')[0]: float(prediction[i]) for i in range(1000)}


imagein = gr.inputs.Image(cast_to="pillow")
label = gr.outputs.Label(num_top_classes=3)

gr.Interface(
    [classify_image_with_mobile_net, classify_image_with_inception_net], 
    imagein, 
    label,
    capture_session=True,
    title="MobileNet vs. ImageNet",
    description="Let's compare 2 state-of-the-art machine learning models that classify images into one of 1,000 categories: MobileNet (top), a lightweight model that has an accuracy of 0.704, vs. InceptionNet (bottom), a much heavier model that has an accuracy of 0.779."
).launch();
