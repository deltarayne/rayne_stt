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
import time 
import webbrowser




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
    time.sleep(2)
    pyautogui.write(text, interval=0.05)  # Adjust the interval as needed for typing speed
    ui.text_field.delete("1.0", tk.END)
    ui.text_field.insert("1.0", text)

def insert_field(target: int):
    insert_text(ui.preset_panels[target].text.rstrip())

def insert_current():
    captured_text = ui.text_field.get("1.0", tk.END).rstrip()
    print(f"about to insert {captured_text}")
    threading.Thread(target=insert_text, args=(captured_text,), daemon=True).start()


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

def test_print():
    print("test!")

def set_new_hotkey():
    pass

class Preset():
    text = ""
    field = None
    def __init__(self, container, field, number: int):
        self.field = field
        self.number = number + 1
        self.panel = tk.Frame(container)
        self.label = tk.Label(self.panel, text = f"Preset:{self.number}")
        self.label.pack(side="left")
        self.load_button = ttk.Button(self.panel, text=f"load {self.number}")
        self.load_button.pack(side="right")
        self.save_button = ttk.Button(self.panel, text=f"save {self.number}")
        self.save_button.pack(side="right")
        
        self.load_button.bind("<Button-1>", self.load_preset)
        self.save_button.bind("<Button-1>", self.save_preset)
        
    def load_preset(self, event):
        self.field.delete("1.0", tk.END)
        self.field.insert("1.0", self.text)
        
    def save_preset(self, event):
        self.text = self.field.get("1.0", tk.END)
        ui.save_presets()
        
class Keybind():
    def __init__(self, name, keystrokes: str, task):
        self.task = task
        self.hotkey = None
        self.change_button = None
        self.combo = None
        self.set_new_hotkey(keystrokes, True)
        self.name = name
        
    def show(self, container):
        self.frm = tk.Frame(container, highlightbackground="black", highlightthickness=1)
        self.name_label = tk.Label(self.frm, text=self.name, width=20, anchor='w')
        self.name_label.pack(side="left", padx=4, pady=2)
        self.change_button = ttk.Button(self.frm, text=self.combo, command=self.receive_key)

        self.change_button.pack(side="right", padx=20)

        self.frm.pack(pady=4)
        
    def receive_key(self): 
        if self.change_button:
            self.change_button.config(state="disabled", text="Press new key combo")
        hotkey_thread = threading.Thread(target=ui.listen_for_hotkey, args=(self,), daemon=True)
        hotkey_thread.start()
        
    def set_new_hotkey(self, combo, initial):
        self.combo = combo
        
        if self.hotkey:
            keyboard.remove_hotkey(self.hotkey)
        self.hotkey = keyboard.add_hotkey(combo, self.task)
        if self.change_button:
            self.change_button.config(state="normal", text=self.combo)
        if initial == False:
            ui.save_keybindings_to_file()
            


class UI(tk.Tk):
    """
    Main application window for the Tkinter program.
    This class inherits from tk.Tk to get all the features of a standard window.
    """
    
    key_bindings = {}
    
    preset_save_file = "presets.json"
    bindings_save_file = "bindings.json"
    
    def __init__(self):
        # Initialize the parent class (tk.Tk)
        super().__init__()
        
        self.title('Rayne STT')
        # --- Window Configuration ---
        # Set the minimum size of the window
        self.minsize(300, 200)
        self.audio_device = sd.default.device[0]
        self.preset_buttons = list[tk.Button]
        # --- Widgets ---
        # Create and place the widgets in the window.
        self.create_widgets()
        self.load_presets()
        #self.set_initial_keybindings()
        self.load_keybindings_from_file()

    def set_initial_keybindings(self, window):
        self.key_bindings['insert_current'] = Keybind("insert current text", 'ctrl+shift+3', insert_current)
        self.key_bindings['start'] = Keybind("start recording", 'ctrl+shift+1', lambda: start_recording(ui.audio_device))
        self.key_bindings['stop'] = Keybind("stop recording", 'ctrl+shift+2', stop_recording)
        self.key_bindings['insert1'] = Keybind("enter preset 1",'ctrl+shift+5', lambda: insert_field(0))
        self.key_bindings['insert2'] = Keybind("enter preset 2",'ctrl+shift+6', lambda: insert_field(1))
        self.key_bindings['insert3'] = Keybind("enter preset 3",'ctrl+shift+7', lambda: insert_field(2))
        self.key_bindings['insert4'] = Keybind("enter preset 4",'ctrl+shift+8', lambda: insert_field(3))
        self.key_bindings['insert5'] = Keybind("enter preset 5",'ctrl+shift+9', lambda: insert_field(4))
        if window:
            window.destroy()
            self.settings()
        self.save_keybindings_to_file()
        self.set_entry_label()
        
    
    def save_keybindings_to_file(self): 
        output = {}
        with open(self.bindings_save_file, 'w') as f:
            for i in self.key_bindings.values():
                print(f"name:{i.name} combo:{i.combo}")
                output.update({i.name : i.combo})
            json.dump(output, f)  
    
    def load_keybindings_from_file(self):
        try: 
            with open(self.bindings_save_file, 'r') as f:
                data = json.load(f)
            self.key_bindings['insert_current'] = Keybind("insert current text", data["insert current text"], insert_current)
            self.key_bindings['start'] = Keybind("start recording", data["start recording"], lambda: start_recording(ui.audio_device))
            self.key_bindings['stop'] = Keybind("stop recording", data["stop recording"], stop_recording)
            self.key_bindings['insert1'] = Keybind("enter preset 1",data["enter preset 1"], lambda: insert_field(0))
            self.key_bindings['insert2'] = Keybind("enter preset 2",data["enter preset 2"], lambda: insert_field(1))
            self.key_bindings['insert3'] = Keybind("enter preset 3",data["enter preset 3"], lambda: insert_field(2))
            self.key_bindings['insert4'] = Keybind("enter preset 4",data["enter preset 4"], lambda: insert_field(3))
            self.key_bindings['insert5'] = Keybind("enter preset 5",data["enter preset 5"], lambda: insert_field(4))
        except FileNotFoundError:
            self.set_initial_keybindings(None)
        self.set_entry_label()

        
    def listen_for_hotkey(self, binding):
        """(Runs in a separate thread)
        Waits for a key combination and schedules the update in the main thread.
        """
        # This is a blocking call; it will wait here until a hotkey is entered.
        hotkey_combo = keyboard.read_hotkey(suppress=False)
        
        # Safely schedule the GUI update and hotkey registration in the main thread.
        self.after(0, binding.set_new_hotkey, hotkey_combo, False)

    def set_entry_label(self):
        self.text_field_bottom_label.configure(text=f"Click in box to edit. Press {self.key_bindings['insert_current'].combo} to type in focused window or {self.key_bindings['start'].combo} to record new.")
        
    def create_widgets(self):
        """
        This method is responsible for creating and arranging all the widgets
        in the main window.
        """

        #main content frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True)

        #name_label
        main_name_label = ttk.Label(
            main_frame,
            text="Rayne STT",
            font=("Bahnschrift", 16, "bold")
        )
    
        main_name_label.pack(pady=10) 
        
        self.start_string = tk.StringVar()
        self.start_string.set("default")
        
        source_panel = tk.Canvas(main_frame)
        source = tk.Label(source_panel, text="Select audio input source:")
        source.pack()
        settings =  ("default", "A", "B", "C")
        self.ddmenu = ttk.Combobox(source_panel, textvariable = self.start_string, values = settings, state="readonly")
        self.ddmenu.pack(side="left")
        self.ddmenu.bind("<<ComboboxSelected>>", self.on_selected)
        source_panel.pack()
        
        self.preset_frame = ttk.Frame(main_frame)
        self.preset_frame.pack(pady=14)

        # Add widgets to the frame
        label = tk.Label(self.preset_frame, text="Presets")
        label.pack(side="top", pady=5) 
        #canvas1 = tk.Canvas(preset_frame)
        
        #scrollbar1 = ttk.Scrollbar(preset_frame, orient='vertical', command=canvas1.yview)
        #canvas1.configure(yscrollcommand=scrollbar1.set)
        #scrollable_frame = ttk.Frame(canvas1)
        
        #scrollbar1.pack(side="right", fill="y")
        #canvas1.pack(side="left")
        self.entry_frame = tk.Frame(main_frame)
        self.text_field_top_label = tk.Label(self.entry_frame, text=f"Current text:")
        self.text_field_bottom_label = tk.Label(self.entry_frame, wraplength=250)
        self.text_field = tk.Text(self.entry_frame, height=5, width=40)
        self.entry_frame.pack(side="bottom", pady=8)
        self.text_field_top_label.pack(pady=8, side="top")
        self.text_field_bottom_label.pack(pady=8, side="bottom")
        self.text_field.pack(pady=8, side="bottom")
        
        self.preset_panels = [] 
        for i in range(5):
            self.preset_panels.insert(len(self.preset_panels), Preset(self.preset_frame, self.text_field, i))
        for i in range(len(self.preset_panels)):
            self.preset_panels[i].panel.pack(padx=5)
        padding_label = tk.Label(self.preset_frame)
        padding_label.pack()
        self.preset_frame.configure(relief=tk.GROOVE)
        menu_bar = tk.Menu(main_frame)
        self.config(menu=menu_bar)
        
        file_list = tk.Menu(menu_bar, tearoff=0)
        edit_list = tk.Menu(menu_bar, tearoff=0)
        help_list = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_list)
        menu_bar.add_cascade(label="Edit", menu=edit_list)
        menu_bar.add_cascade(label="Help", menu=help_list)
        file_list.add_command(label="Quit", command=self.destroy)
        edit_list.add_command(label="Keybindings...", command=self.settings)
        edit_list.add_separator()
        edit_list.add_command(label="Copy", command=self.copy_to_clipboard)
        edit_list.add_command(label="Paste", command=self.paste_from_clipboard)
        help_list.add_command(label="Help", command=lambda:webbrowser.open_new_tab("https://github.com/deltarayne/rayne_stt"))
        help_list.add_command(label="About", command=self.about)
            
        #testlabel = tk.Label(preset_frame, text="test label")
        #testlabel.pack()
    def copy_to_clipboard(self):
        self.clipboard_clear()
        self.clipboard_append(self.text_field.get("1.0", tk.END))
    
    def paste_from_clipboard(self):
        text = self.clipboard_get() 
        self.text_field.delete("1.0", tk.END)
        self.text_field.insert("1.0", text)

    def about(self):
        about_window = tk.Toplevel(self)
        about_button_frame= tk.Frame(about_window)
        about_window.title("About")
        about_name_label = tk.Label(about_window, text="Rayne STT by Miranda R.", font=("Bahnschrift", 16, "bold"))
        about_info_label = tk.Label(about_window, text="This software uses OpenAI's Whisper V3 model for speech recognition and is distributed at no cost, under the terms of the GPL. (Full license text and details available in ReadMe)", wraplength=300)
        about_name_label.pack(pady=16, padx=20)
        about_info_label.pack(pady=16, padx=20)
        about_github_button = ttk.Button(about_button_frame, text="Github", command=lambda: webbrowser.open_new_tab("https://github.com/deltarayne/rayne_stt"))
        about_close_button = ttk.Button(about_button_frame, text="Close", command=about_window.destroy) 
        about_github_button.pack(pady=4)
        about_close_button.pack(pady=4)
        about_button_frame.pack(pady=16)
        #     label = tk.Label(preset_panels[i], text=f"hello{i}")
        #     save_button = ttk.Button(preset_panels[i], text="save")
        #     use_button = ttk.Button(preset_panels[i], text="use")
        #     save_button.pack(side="right")
        #     use_button.pack(side="right")
        #     label.pack()
        #     preset_panels[i].pack()
            
        #canvas1.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # scrollable_frame.bind(
        #     "<Configure>",
        #     lambda e: canvas1.configure(
        #     scrollregion=canvas1.bbox("all")
        #     )
        # )
    def settings(self):
        settings_window = tk.Toplevel(self)
        settings_window.title("Settings")
        bind_frame = tk.Frame(settings_window)
        for i in self.key_bindings.values():
            i.show(bind_frame)
        close_button = tk.Button(settings_window, text="Close", command=settings_window.destroy)
        defaults_button = tk.Button(settings_window, text="Reset to Defaults", command=lambda: self.set_initial_keybindings(settings_window))  
        bind_frame.pack(pady=12, padx=12)
        defaults_button.pack(pady=10, side="bottom")
        close_button.pack(pady=5, side="bottom")
            
    def save_presets(self):
        output = {}
        with open(self.preset_save_file, 'w') as f:
            for i in range(len(self.preset_panels)):
                output.update({i: self.preset_panels[i].text})
            json.dump(output, f)    
            
    def load_presets(self):
        input = {}
        try:
            with open(self.preset_save_file, 'r') as f:
                data = json.load(f)
                print('__data___')
                print(data)
            for i in range(len(self.preset_panels)):
                self.preset_panels[i].text = data[str(i)]
        except FileNotFoundError:
            pass    
        
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
        # start_key = 'ctrl+shift+1'
        # stop_key = 'ctrl+shift+2'
        # keyboard.remove_all_hotkeys()
        # keyboard.add_hotkey(start_key, start_recording, (ui.audio_device,))
        # keyboard.add_hotkey(stop_key, stop_recording)
        
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
    insert_current_text = 'ctrl+shift+3'
    insert1 = 'ctrl+shift+5'
    insert2 = 'ctrl+shift+6'
    insert3 = 'ctrl+shift+7'
    insert4 = 'ctrl+shift+8'
    insert5 = 'ctrl+shift+9'



    # Keep the script alive to listen for hotkeys.
    #keyboard.wait('esc')

    # run the main loop with tkinter
    ui = UI()
    dlist = sd.query_devices()
    ui.populate_device_menu(dlist)

    
    # start_hotkey = keyboard.add_hotkey(start_key, start_recording, (ui.audio_device,))
    # stop_hotkey = keyboard.add_hotkey(stop_key, stop_recording)
    # insert_current_hotkey = keyboard.add_hotkey(insert_current_text, insert_current)
    # insert1_hotkey = keyboard.add_hotkey(insert1, insert_field, args=(0,))
    # insert2_hotkey = keyboard.add_hotkey(insert2, insert_field, args=(1,))
    # insert3_hotkey = keyboard.add_hotkey(insert3, insert_field, args=(2,))
    # insert4_hotkey = keyboard.add_hotkey(insert4, insert_field, args=(3,))
    # insert5_hotkey = keyboard.add_hotkey(insert5, insert_field, args=(4,))

    # global_hotkeys = {
    #     'start' : start_hotkey,
    #     'stop' : stop_hotkey,
    #     'insert_current' : insert_current_hotkey,
    #     'insert1' : insert1_hotkey, 
    #     'insert2' : insert2_hotkey, 
    #     'insert3' : insert3_hotkey, 
    #     'insert4' : insert4_hotkey,
    #     'insert5' : insert5_hotkey
    # }
    
    # ui.hotkeys = global_hotkeys

    print(f"Press '{start_key}' to start recording.")
    print("Press 'Esc' to exit the program.")
    ui.mainloop()

    # --- Cleanup ---
    print("\nExiting program. Cleaning up hotkeys...")
    keyboard.remove_all_hotkeys()
    print("Cleanup complete.")




