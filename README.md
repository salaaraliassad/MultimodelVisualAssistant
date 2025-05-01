MultimodalVisionAssistant is a conversational AI system built to support visually impaired users. By combining state-of-the-art models for object detection, depth estimation, speech recognition, visual question answering (VQA), and text-to-speech, this assistant provides real-time, accessible, and context-aware responses to users' spoken questions about their environment. The app can be found on Hugging Face Space using the following link: https://huggingface.co/spaces/saa231/MutimodalVisionAssistant 
---

**Features**
  1. Object detection using YOLOv9
  2. Depth estimation with MiDaS (Intel/dpt-large)
  3. Audio-to-text transcription using Whisper (openai/whisper-small)
  4. Visual Question Answering with Gemma 3 (google/gemma-3-4b-it)
  5. Shared visual context generation from scene understanding
  6. Text-to-speech using Tacotron2 (tts_models/en/ljspeech/tacotron2-DDC)

title: MutimodalVisionAssistant
emoji: 🏢
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
pinned: false
license: mit
short_description: Multimodal AI for Visual Impairment Support
models:
- Intel/dpt-large
- openai/whisper-small
- google/gemma-3-4b-it
- yolov9c.pt
- tts_models/en/ljspeech/tacotron2-DDC
---

**License**
This project is licensed under the MIT License.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
