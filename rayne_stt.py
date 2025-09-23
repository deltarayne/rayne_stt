import keyboard
import pyautogui
import sounddevice as sd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
import json



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
GUI_DEBUG = False
# --- Model and Pipeline Setup ---
if GUI_DEBUG == False:
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
    ui.text_field.delete("1.0", tk.END)
    ui.text_field.insert("1.0", text)
    
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


class Preset():
    text = ""
    field = None
    def __init__(self, container, field, number: int):
        self.field = field
        self.number = number
        self.panel = tk.Canvas(container)
        self.label = tk.Label(self.panel, text = f"Preset:{number}")
        self.label.pack(side="left")
        self.load_button = ttk.Button(self.panel, text=f"load{number}")
        self.load_button.pack(side="right")
        self.save_button = ttk.Button(self.panel, text=f"save{number}")
        self.save_button.pack(side="right")
        
        self.load_button.bind("<Button-1>", self.load_preset)
        self.save_button.bind("<Button-1>", self.save_preset)
        
    def load_preset(self, event):
        self.field.delete("1.0", tk.END)
        self.field.insert("1.0", self.text)
        
    def save_preset(self, event):
        self.text = self.field.get("1.0", tk.END)
        ui.save_presets()
        
    
class UI(tk.Tk):
    """
    Main application window for the Tkinter program.
    This class inherits from tk.Tk to get all the features of a standard window.
    """
    
    save_file = "presets.json"
    def __init__(self):
        # Initialize the parent class (tk.Tk)
        super().__init__()

        # --- Window Configuration ---
        self.title("Simple Tkinter App")
        # Set the initial size of the window (width x height)
        #self.geometry("500x500")
        # Set the minimum size of the window
        self.minsize(300, 200)
        self.audio_device = sd.default.device[0]
        self.preset_buttons = list[tk.Button]
        # --- Widgets ---
        # Create and place the widgets in the window.
        self.create_widgets()
        self.load_presets()


    def create_widgets(self):
        """
        This method is responsible for creating and arranging all the widgets
        in the main window.
        """
        # Create a frame to hold the content.
        # Using a frame is good practice for organizing widgets.
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True)

        # Create a label widget
        welcome_label = ttk.Label(
            main_frame,
            text="David's Audio Input Program",
            font=("Bahnschrift", 16, "bold")
        )
        # The pack() geometry manager places the widget in the window.
        welcome_label.pack(pady=10) # pady adds vertical padding
        
        self.start_string = tk.StringVar()
        self.start_string.set("default")
        
        source_panel = tk.Canvas(main_frame)
        source = tk.Label(source_panel, text="Select audio input source:")
        source.pack()
        settings =  ("default", "A", "B", "C")
        self.ddmenu = ttk.Combobox(source_panel, textvariable = self.start_string, values = settings, state="readonly")
        #ddmenu.current(0)
        self.ddmenu.pack(side="left")
        self.ddmenu.bind("<<ComboboxSelected>>", self.on_selected)
        source_panel.pack()
        
        my_frame = ttk.Frame(main_frame)
        my_frame.pack() # Or use .grid() or .place() to position the frame

        # Add widgets to the frame
        label = tk.Label(my_frame, text="Presets")
        label.pack(side="top", pady=5) 
        canvas1 = tk.Canvas(my_frame, background='gray75', width=300)
        
        scrollbar1 = ttk.Scrollbar(my_frame, orient='vertical', command=canvas1.yview)
        canvas1.configure(yscrollcommand=scrollbar1.set)
        scrollable_frame = ttk.Frame(canvas1)
        
        scrollbar1.pack(side="right", fill="y")
        canvas1.pack(side="left")
    
        self.text_field = tk.Text(main_frame, height=5, width=20)
        
        
        self.preset_panels = []
        for i in range(5):
            self.preset_panels.insert(len(self.preset_panels), Preset(scrollable_frame, self.text_field, i))
        for i in range(len(self.preset_panels)):
            self.preset_panels[i].panel.pack()
            
        type_this = ttk.Button(main_frame, text="type this")
        type_this.pack(side="right")
        self.text_field.pack(side="right")
        
        
            
        #testlabel = tk.Label(my_frame, text="test label")
        #testlabel.pack()

            
        #     label = tk.Label(preset_panels[i], text=f"hello{i}")
        #     save_button = ttk.Button(preset_panels[i], text="save")
        #     use_button = ttk.Button(preset_panels[i], text="use")
        #     save_button.pack(side="right")
        #     use_button.pack(side="right")
        #     label.pack()
        #     preset_panels[i].pack()
            
        canvas1.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas1.configure(
            scrollregion=canvas1.bbox("all")
            )
        )
        
    def save_presets(self):
        output = {}
        with open(self.save_file, 'w') as f:
            for i in range(len(self.preset_panels)):
                output.update({i: self.preset_panels[i].text})
            json.dump(output, f)    
            
    def load_presets(self):
        input = {}
        with open(self.save_file, 'r') as f:
            data = json.load(f)
            print('__data___')
            print(data)
        for i in range(len(self.preset_panels)):
            self.preset_panels[i].text = data[str(i)]
            
    def on_selected(self, event):
        print(f"Selected value is: {self.ddmenu.get()}")
        target = self.ddmenu.get()
        for i in self.dlist:
            if i['name'] == target:
                target_index = i['index']
                break
        self.set_audio_device(target_index)
     
        
    # --- Event Handlers ---
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
