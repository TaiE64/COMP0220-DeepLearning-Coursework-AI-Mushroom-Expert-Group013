import gradio as gr
import os
import sys
from pathlib import Path
import time
import speech_recognition as sr
from gtts import gTTS
import tempfile

# Add the current directory to sys.path to ensure imports work
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Define Image Directory
PROJECT_ROOT = current_dir.parent.parent
IMAGE_DIR = PROJECT_ROOT / "front_end" / "pictures"

if not IMAGE_DIR.exists():
    print(f"Warning: Image directory not found at {IMAGE_DIR}, creating it...")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Import the existing chatbot class
try:
    from chat_with_image_integrated import MushroomChatbot
except ImportError:
    sys.path.append(str(current_dir / "LLM Core" / "main_script"))
    from chat_with_image_integrated import MushroomChatbot

def create_demo():
    # Initialize the chatbot
    print("Initializing Mushroom Assistant...")
    bot = MushroomChatbot(use_rag=True, enable_image=True)
    
    # Textures / Avatars
    user_avatar = str(IMAGE_DIR / "user_avatar.png")
    bot_avatar = str(IMAGE_DIR / "mushroom_mario.png")
    
    # Check Avatars (Same as before)
    if not os.path.exists(user_avatar):
        try:
            from PIL import Image, ImageDraw
            def create_avatar(filename, text, color):
                img = Image.new('RGB', (100, 100), color=color)
                d = ImageDraw.Draw(img)
                d.text((25, 25), text, fill=(255, 255, 255))
                img.save(filename)
            create_avatar(user_avatar, "User", "#555555")
        except Exception:
            user_avatar = None

    # --- Audio Processing Functions ---
    def transcribe(audio_path):
        """Convert voice to text using SpeechRecognition"""
        if audio_path is None:
            return ""
        
        r = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data) # Uses Google Web API (Online)
                print(f"Recognized speech: {text}")
                return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"
        except Exception as e:
            return f"Error: {e}"

    import re

    def clean_text_for_tts(text):
        """Remove Markdown formatting for better speech synthesis"""
        # Remove bold/italic markers (* or _)
        text = re.sub(r'[\*_]{1,3}', '', text)
        # Remove code blocks
        text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
        # Remove links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove headers (#)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Remove bullet points at start of lines
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def text_to_speech(text):
        """Convert text to speech using gTTS"""
        if not text:
            return None
        
        # Clean text first
        clean_text = clean_text_for_tts(text)
        if not clean_text:
            return None
            
        try:
            # Create a temporary file
            fd, path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            
            # Generate speech
            tts = gTTS(text=clean_text, lang='en')
            tts.save(path)
            return path
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    # --- UI Layout ---
    with gr.Blocks(title="🍄 Integrated Mushroom Assistant") as demo:
        
        gr.Markdown(
            """
            # 🍄 Integrated Mushroom Assistant (Voice Enabled 🎙️)
            ### AI-Powered Identification & Knowledge Base
            """
        )
        
        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="filepath", 
                    label="Upload Mushroom Image", 
                    sources=["upload", "clipboard"],
                    height=250
                )
                
                # Audio Input
                audio_input = gr.Audio(
                    sources=["microphone"], 
                    type="filepath",
                    label="Speak to Assistant 🎙️"
                )
                
                with gr.Accordion("Settings", open=False):
                    rag_toggle = gr.Checkbox(label="Enable RAG", value=True)
                    clear_btn = gr.Button("🗑️ Clear History")

            # Right Column: Chat + Audio Output
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=450,
                    avatar_images=(user_avatar, bot_avatar),
                )
                
                # Audio Output (Assistant's Voice)
                audio_output = gr.Audio(
                    label="Assistant Voice 🔊", 
                    autoplay=True, 
                    type="filepath",
                    interactive=False
                )
                
                msg_input = gr.Textbox(
                    placeholder="Type or speak...",
                    label="Your Message",
                    autofocus=True
                )
                submit_btn = gr.Button("Submit", variant="primary")

        # States
        last_image_state = gr.State(value=None)
        action_queue = gr.State(value=[]) 

        # --- Interaction Logic ---
        def clear_memory():
            bot.clear_history()
            return None, [], None

        def handle_user_input(text_msg, audio_path, history, img_path, last_img_path):
            """Step 1: Process user input (Text OR Audio) -> Update UI -> Queue Bot Action"""
            if history is None: history = []
            if len(history) == 0: bot.clear_history()

            user_text = text_msg
            
            # 1. If Audio is provided, transcribe it first
            if audio_path:
                print("Transcribing audio...")
                transcribed_text = transcribe(audio_path)
                if transcribed_text:
                    user_text = transcribed_text 
            
            actions = []
            current_img_path = last_img_path

            # 2. Handle New Image
            if img_path and img_path != last_img_path:
                history.append({"role": "user", "content": f"![](/file={img_path})"})
                actions.append(f"@image {img_path}")
                current_img_path = img_path
            
            # 3. Handle Text (from typing or transcription)
            if user_text:
                history.append({"role": "user", "content": user_text})
                actions.append(user_text)

            # Return: 
            # Clear text input, Clear audio input (so it doesn't resend), Update Chat, Update Image State, Update Action Queue
            return "", None, history, current_img_path, actions

        def generate_bot_response(history, actions):
            """Step 2: Bot thinks and replies -> Returns Text History AND Audio Path"""
            if not actions:
                return history, None
            
            full_response_text = ""
            
            for action in actions:
                response = bot.chat(action)
                history.append({"role": "assistant", "content": response})
                full_response_text += response + " "
            
            # Convert the final response to audio
            print("Generating TTS...")
            audio_file = text_to_speech(full_response_text)
            
            return history, audio_file

        # --- Wiring Events ---
        
        # 1. Submit Text
        msg_input.submit(
            handle_user_input,
            inputs=[msg_input, audio_input, chatbot, image_input, last_image_state],
            outputs=[msg_input, audio_input, chatbot, last_image_state, action_queue]
        ).then(
            generate_bot_response,
            inputs=[chatbot, action_queue],
            outputs=[chatbot, audio_output]
        )
        
        # 2. Submit Button
        submit_btn.click(
            handle_user_input,
            inputs=[msg_input, audio_input, chatbot, image_input, last_image_state],
            outputs=[msg_input, audio_input, chatbot, last_image_state, action_queue]
        ).then(
            generate_bot_response,
            inputs=[chatbot, action_queue],
            outputs=[chatbot, audio_output]
        )
        
        
        # 3. Audio Input Change (Auto-submit when recording stops)
        audio_input.stop_recording(
            handle_user_input,
            inputs=[msg_input, audio_input, chatbot, image_input, last_image_state],
            outputs=[msg_input, audio_input, chatbot, last_image_state, action_queue]
        ).then(
            generate_bot_response,
            inputs=[chatbot, action_queue],
            outputs=[chatbot, audio_output]
        )

        # Clear
        clear_btn.click(clear_memory, outputs=[image_input, chatbot, last_image_state])
        
        # RAG
        rag_toggle.change(lambda x: setattr(bot, 'use_rag', x), inputs=[rag_toggle], outputs=[])

    return demo

if __name__ == "__main__":
    demo = create_demo()
    print("Launching Voice-Enabled Interface...")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
