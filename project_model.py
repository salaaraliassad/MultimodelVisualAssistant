# project_module.py

# Import libraries for ML, CV, NLP, audio, and TTS
import torch, cv2, os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline, DPTFeatureExtractor, DPTForDepthEstimation
from TTS.api import TTS
from huggingface_hub import login

# Authenticate to Hugging Face using environment token
login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])

# Set device for computation (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load all models
yolo_model = YOLO("yolov9c.pt")  # YOLOv9 for object detection
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device).eval()  # MiDaS for depth
depth_feat = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")  # Feature extractor for depth model

# Whisper for audio transcription
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=0 if torch.cuda.is_available() else -1
)

# GEMMA for image+text to text QA
gemma_pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

# Text-to-speech
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# -------------------------------
# Session Management Class
# -------------------------------

class VisualQAState:
    """
    Stores the current image context and chat history for follow-up questions.
    """
    def __init__(self):
        self.current_image: Image.Image = None
        self.annotated_image: Image.Image = None
        self.visual_context: str = ""
        self.message_history = []

    def reset(self, image: Image.Image, annotated_image: Image.Image, visual_context: str):
        """
        Called when a new image is uploaded.
        Resets context and starts new message history.
        """
        self.current_image = image
        self.annotated_image = annotated_image
        self.visual_context = visual_context
        self.message_history = [
            {
                "role": "system",  # System prompt
                "content": (
                    "You are a helpful visual assistant designed for visually impaired users that assists users by answering their questions. "
                    #"You must provide detailed, descriptive, and spatially-aware answers based on the given image, the question asked, and conversation history. "
                    #"Always describe what you see clearly and help the user understand the scene. "
                    'If unsure, say "I am not certain."'
                )
            },
            {
                "role": "user",  # The user input
                "content": [
                    {"type": "image", "image": self.current_image},  # Image context
                    {"type": "text", "text": self.visual_context}   # Visual context description
                ]
            }
        ]

    def add_question(self, question: str):
        """
        Adds a follow-up question only if the last message was from assistant.
        Ensures alternating user/assistant messages.
        """
        if not self.message_history or self.message_history[-1]["role"] == "assistant":
            self.message_history.append({
                "role": "user",
                "content": [{"type": "text", "text": question}]
            })

    def add_answer(self, answer: str):
        """
        Appends the assistant's response to the conversation history.
        """
        self.message_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer}]
        })

# -------------------------------
# Generate Context from Image
# -------------------------------

def generate_visual_context(pil_image: Image.Image):
    # Convert to OpenCV and RGB formats
    rgb_image = np.array(pil_image)
    cv2_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Object detection
    yolo_results = yolo_model.predict(cv2_image)[0]
    boxes = yolo_results.boxes
    class_names = yolo_model.names

    # Depth estimation
    depth_inputs = depth_feat(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        depth_output = depth_model(**depth_inputs)
    depth_map = depth_output.predicted_depth.squeeze().cpu().numpy()
    depth_map_resized = cv2.resize(depth_map, (rgb_image.shape[1], rgb_image.shape[0]))

    # Draw bounding boxes on a copy of the image
    annotated_image = cv2_image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = class_names[int(box.cls[0])]
        conf = float(box.conf[0])

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Extract context
    shared_visual_context = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = class_names[int(box.cls[0])]
        conf = float(box.conf[0])

        depth_crop = depth_map_resized[y1:y2, x1:x2]
        avg_depth = float(depth_crop.mean()) if depth_crop.size > 0 else None

        x_center = (x1 + x2) / 2
        pos = "left" if x_center < rgb_image.shape[1] / 3 else "right" if x_center > 2 * rgb_image.shape[1] / 3 else "center"

        shared_visual_context.append({
            "label": label,
            "confidence": conf,
            "avg_depth": avg_depth,
            "position": pos
        })

    descriptions = []
    for obj in shared_visual_context:
        d = f"{obj['avg_depth']:.1f} units" if obj["avg_depth"] else "unknown"
        s = obj.get("position", "unknown")
        c = obj.get("confidence", 0.0)
        descriptions.append(f"a {obj['label']} ({c:.2f} confidence) is at {d} on the {s}")

    context_sentence = "In the image, " + ", ".join(descriptions) + "."

    # Save annotated image
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    return context_sentence, annotated_pil


# -------------------------------
# Main Multimodal Processing Function
# -------------------------------

# Create a global session object to persist across follow-ups
session = VisualQAState()

def process_inputs(
    session: VisualQAState,
    image: Image.Image = None,
    question: str = "",
    audio_path: str = None,
    enable_tts: bool = True
):
    if image:
        # Generate visual context and annotated image
        visual_context, annotated_image = generate_visual_context(image)
        
        # Reset session with the current image and visual context
        session.reset(image, annotated_image, visual_context)

    if audio_path:
        # Process audio to text
        audio_text = whisper_pipe(audio_path)["text"]
        question += ' ' + audio_text.strip()

    # Add user's new question to the history
    session.add_question(question)


    vqa_prompt = "You are a helpful visual assistant designed for visually impaired users that assists users by answering their questions. Answer the following question with the help of the shared visual context: " + question + "Shared visual context: " + visual_context 
    
    # Gemma prompt input
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": session.current_image},
            {"type": "text", "text": vqa_prompt}]
    }]
    
    # Call to gemma_pipe
    gemma_output = gemma_pipe(text=messages, max_new_tokens=500)

    # Handle the output from Gemma model safely
    if isinstance(gemma_output, list) and len(gemma_output) > 0:
        gemma_text = gemma_output[0]["generated_text"][-1]["content"]
    if isinstance(gemma_text, str):
        answer = gemma_text
    else:
        answer = "No valid output from Gemma model."

    # Save assistant's answer into session history
    session.add_answer(answer)

    # Text-to-speech output
    output_audio_path = "response.wav"
    if enable_tts:
        tts.tts_to_file(text=answer, file_path=output_audio_path)
    else:
        output_audio_path = None

    return answer, output_audio_path
