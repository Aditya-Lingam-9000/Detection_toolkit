# app.py
import gradio as gr
from main import detect_objects, AVAILABLE_MODELS

demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(
            choices=list(AVAILABLE_MODELS.keys()),
            value=list(AVAILABLE_MODELS.keys())[0],
            label="Select YOLOv11 Model"
        )
    ],
    outputs=gr.Image(type="pil"),
    title="YOLOv11 Object Detection"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
