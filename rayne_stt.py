import keyboard
import pyautogui
import sounddevice as sd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk

# --- Configuration ---
# The recording sample rate must match what the Whisper model was trained on.
RATE = 16000
# Use a smaller chunk size for lower latency.
CHUNK_SIZE = 1024
# Set device for PyTorch    
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
audio_input_device = sd.default.device
# --- State Management for Recording ---
# These global variables will manage the recording state across different threads.
is_recording = False
audio_data = []

# --- Model and Pipeline Setup ---
print("Loading the Whisper model. This may take a moment...")
model_id = "openai/whisper-large-v3"

try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True
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
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have a stable internet connection and necessary dependencies installed.")
    exit()

def record_audio(device_index):
    """
    This function is run in a separate thread. It continuously reads from the
    audio stream and adds the data to our `audio_data` list.
    """
    global is_recording, audio_data
    
    device_id = 2
    print(f"now recording on device: {device_index}")
    # We use sd.InputStream to continuously capture audio data without blocking.
    with sd.InputStream(device=device_index, samplerate=RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE) as stream:
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
    
def start_recording(device_index):
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
    threading.Thread(target=record_audio, args=(device_index,), daemon=True).start()

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

class UI(tk.Tk):
    """
    Main application window for the Tkinter program.
    This class inherits from tk.Tk to get all the features of a standard window.
    """
    def __init__(self):
        # Initialize the parent class (tk.Tk)
        super().__init__()

        # --- Window Configuration ---
        self.title("Simple Tkinter App")
        # Set the initial size of the window (width x height)
        self.geometry("500x500")
        # Set the minimum size of the window
        self.minsize(300, 200)
        self.audio_device = sd.default.device[0]
        # --- Widgets ---
        # Create and place the widgets in the window.
        self.create_widgets()

    def create_widgets(self):
        """
        This method is responsible for creating and arranging all the widgets
        in the main window.
        """
        # Create a frame to hold the content.
        # Using a frame is good practice for organizing widgets.
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill="both")

        # Create a label widget
        welcome_label = ttk.Label(
            main_frame,
            text="David's Audio Input Program",
            font=("Helvetica", 16)
        )
        # The pack() geometry manager places the widget in the window.
        welcome_label.pack(pady=10) # pady adds vertical padding

        # Create a button widget
        # The 'command' option links the button to a function.
        action_button = ttk.Button(
            main_frame,
            text="Click Me!",
            command=self.on_button_click
        )
        action_button.pack(pady=10)
        
        self.start_string = tk.StringVar()
        self.start_string.set("default")
        settings =  ("default", "A", "B", "C")
        self.ddmenu = ttk.Combobox(main_frame, textvariable = self.start_string, values = settings, state="readonly")
        #ddmenu.current(0)
        self.ddmenu.pack(pady=50)
        self.ddmenu.bind("<<ComboboxSelected>>", self.on_selected)
        
    def on_selected(self, event):
        print(f"Selected value is: {self.ddmenu.get()}")
        target = self.ddmenu.get()
        for i in self.dlist:
            if i['name'] == target:
                target_index = i['index']
                break
        print(f"index found in values at:{target_index}")
        self.set_audio_device(target_index)
        print(f"device index is now: {self.audio_device}")
        print(f"device query information for selected device:{sd.query_devices(target_index)}")
        
    # --- Event Handlers ---
    def on_button_click(self):
        """
        This function is called when the 'action_button' is clicked.
        """
        print("Button was clicked!")

    def set_audio_device(self, device):
        print(f"setting audio device to {device}")
        self.audio_device = device
        start_key = 'ctrl+shift+1'
        stop_key = 'ctrl+shift+2'
        keyboard.remove_all_hotkeys()
        keyboard.add_hotkey(start_key, start_recording, (ui.audio_device,))
        keyboard.add_hotkey(stop_key, stop_recording)
        
    def populate_device_menu(self, devices: list[dict]):
        names = []
        self.dlist = []
        for i in devices:
            if i != None:
                if i['max_input_channels'] > 0 and not i['name'] in names:
                    names.insert(len(names), i['name'])
                self.dlist.insert(len(dlist), i)
        self.ddmenu['values'] = names


        
    
    
        
if __name__ == "__main__":
    print("--default device--")
    print(sd.default.device)
    print("---input devices---")
    #for i in sd.query_devices():
        #print(i['name'])
    for i in sd.query_devices():
            print(i)
            
    print("__api hosts__")
    print(sd.query_hostapis())
        
    # --- Hotkey Setup ---
    # The `keyboard` library runs callbacks in their own threads, which is perfect
    # for our start/stop logic without blocking the main script.
    start_key = 'ctrl+shift+1'
    stop_key = 'ctrl+shift+2'



    # Keep the script alive to listen for hotkeys.
    #keyboard.wait('esc')

    # run the main loop with tkinter
    ui = UI()
    dlist = sd.query_devices()
    ui.populate_device_menu(dlist)
    
    keyboard.add_hotkey(start_key, start_recording, (ui.audio_device,))
    keyboard.add_hotkey(stop_key, stop_recording)

    print(f"Press '{start_key}' to start recording.")
    print("Press 'Esc' to exit the program.")
    ui.mainloop()

    # --- Cleanup ---
    print("\nExiting program. Cleaning up hotkeys...")
    keyboard.remove_all_hotkeys()
    print("Cleanup complete.")
