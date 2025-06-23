import keyboard
import pyautogui
import sounddevice as sd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import threading

# --- Configuration ---
# The recording sample rate must match what the Whisper model was trained on.
RATE = 16000
# Use a smaller chunk size for lower latency.
CHUNK_SIZE = 1024
# Set device for PyTorch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# --- State Management for Recording ---
# These global variables will manage the recording state across different threads.
is_recording = False
audio_data = []

# --- Model and Pipeline Setup ---
print("Loading the Whisper model. This may take a moment...")
model_id = "openai/whisper-large-v3"

try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have a stable internet connection and necessary dependencies installed.")
    exit()

def record_audio():
    """
    This function is run in a separate thread. It continuously reads from the
    audio stream and adds the data to our `audio_data` list.
    """
    global is_recording, audio_data
    
    # We use sd.InputStream to continuously capture audio data without blocking.
    with sd.InputStream(samplerate=RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE) as stream:
        print("Recording started... Press Ctrl+Shift+2 to stop.")
        while is_recording:
            # Read a chunk of audio.
            chunk, overflowed = stream.read(CHUNK_SIZE)
            if overflowed:
                print("Warning: Audio input overflowed!")
            audio_data.append(chunk)
            
def transcribe_audio():
    """
    This function takes the recorded audio data, processes it, and
    runs it through the transcription pipeline.
    """
    global audio_data
    print("Recording stopped. Preparing for transcription...")

    if not audio_data:
        print("No audio was recorded.")
        return

    # Concatenate all the recorded chunks into a single NumPy array.
    full_audio = np.concatenate(audio_data, axis=0)
    # The pipeline expects a 1D array, so we squeeze the channel dimension.
    audio_input_squeezed = full_audio.squeeze()

    print("Transcribing... This may take a moment.")
    try:
        # Pass the raw audio data and sampling rate to the pipeline.
        result = pipe({"sampling_rate": RATE, "raw": audio_input_squeezed})
        print("\n--- Transcription ---")
        print(result["text"].strip())
        insert_text(result["text"].strip())
        print("---------------------\n")
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
    
    # Clear the audio data for the next recording
    audio_data = []
    print("Ready to record again. Press Ctrl+Shift+1 to start.")

def insert_text(text):
    pyautogui.write(text, interval=0.05)  # Adjust the interval as needed for typing speed
    
def start_recording():
    """
    Callback function for the 'start' hotkey.
    It sets the recording flag and starts the recording thread.
    """
    global is_recording
    if is_recording:
        print("Already recording.")
        return
        
    is_recording = True
    # Start the recording in a separate thread to keep the main thread responsive.
    threading.Thread(target=record_audio, daemon=True).start()

def stop_recording():
    """
    Callback function for the 'stop' hotkey.
    It clears the recording flag and starts the transcription process.
    """
    global is_recording
    if not is_recording:
        return
        
    is_recording = False
    # The transcription can be slow, so run it in a thread to not block the hotkey listener.
    threading.Thread(target=transcribe_audio, daemon=True).start()

if __name__ == "__main__":
    # --- Hotkey Setup ---
    # The `keyboard` library runs callbacks in their own threads, which is perfect
    # for our start/stop logic without blocking the main script.
    start_key = 'ctrl+shift+1'
    stop_key = 'ctrl+shift+2'

    keyboard.add_hotkey(start_key, start_recording)
    keyboard.add_hotkey(stop_key, stop_recording)

    print(f"Press '{start_key}' to start recording.")
    print("Press 'Esc' to exit the program.")

    # Keep the script alive to listen for hotkeys.
    keyboard.wait('esc')

    # --- Cleanup ---
    print("\nExiting program. Cleaning up hotkeys...")
    keyboard.remove_all_hotkeys()
    print("Cleanup complete.")
