import gradio as gr
from predict import detector_function as detector
from PIL import Image


def detect_objects(image):
    image_path = 'uploaded_image.jpg'
    image.save(image_path)
    output_image_path = detector(image_path=image_path)

    return Image.open(output_image_path)

demo = gr.Interface(
    fn = detect_objects,
    inputs = gr.Image(type='pil'),
    outputs = gr.Image(label="Output Image with Detections"),
    title ="Object Detection Gradio App with fasterrcnn",
    description="Upload an image to detect objects using a pretrained Faster R-CNN model."
)

if __name__ == "__main__":
    demo.launch()