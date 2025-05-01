from PIL import Image
import gradio as gr
import uuid
from project_model import process_inputs, session


def handle_question(image, audio):
    try:
        # No input provided
        if not image and not audio:
            return "Please upload an image and/or record an audio clip.", None

        # New question with both image + audio
        if image is not None:
            session.current_image = image
            session.messages = []
            session.images = []

        if session.current_image is None:
            return "No initial image found. Please upload an image first.", None

        # Process inputs
        message, answer_audio = process_inputs(session, image=session.current_image, audio_path=audio)

        # Save images (only if a new image was uploaded)
        if image:
            unique_id = uuid.uuid4().hex
            original_path = f"uploaded_image_{unique_id}.png"
            annotated_path = f"annotated_image_{unique_id}.png"
            image.save(original_path)

            if session.annotated_image:
                session.annotated_image.save(annotated_path)

        # Build Markdown reply
        markdown_reply = f"**{message}**\n\n"
        # Uncomment if you want images/audio previews inside Markdown
        # markdown_reply += f"![Original Image](file/{original_path})\n\n"
        # markdown_reply += f"![Annotated Image](file/{annotated_path})\n\n"
        # markdown_reply += f"<audio controls autoplay><source src='file/{answer_audio}' type='audio/wav'></audio>"

        return markdown_reply, answer_audio

    except ValueError as e:
        return f"Error: {str(e)}", None


# --- Gradio App ---
with gr.Blocks() as demo:
    gr.Markdown("## Multimodal Visual Q&A with Audio Output")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload or Capture Image", sources=["upload", "webcam"], type="pil")
            audio_input = gr.Audio(label="Ask a Question (Voice)", sources=["microphone"], type="filepath")
            submit_btn = gr.Button("Submit Question")

        with gr.Column():
            status_output = gr.Markdown(label="Response")  # Use Markdown to format answers
            audio_output = gr.Audio(label="Audio Answer", interactive=False)

    # Connect button to function
    submit_btn.click(
        fn=handle_question,
        inputs=[image_input, audio_input],
        outputs=[status_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
