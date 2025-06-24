
# rayne_stt

## Usage

See installation guide and requirements below. 

download rayne_stt.py and run it from the command line:

`python rayne_stt.py`

the program takes a moment to start and if you haven't used the specific whisper model in this environment before, the transformers module will download it (it only needs to do this once). Once the model is downloaded the program will prompt you to press a hotkey combo to begin recording. If you want to type in a specific text field or window, make sure you're focused on that window (you've clicked in the window or text field and have a blinking cursor) Pressing the stop hotkey (should also show up in the command line) will stop recording and the program will begin to process your output and automatically type it into whatever field or window has focus. 

## Installation

### install required python modules

`pip install transformers pyautogui keyboard`

### install torch

use the website for torch:

[torch](https://pytorch.org)

installing pytorch for the correct CUDA version will insure the GPU is available to the whisper model. This isn't required but the program will operate considerably slower without the GPU. 

to select a version compatible with your OS/CUDA version, use the widget on the torch website to get the proper pip3 command and paste the output into your command line.
*torch for CUDA 12.8 is compatible with all versions 12.x higher than 8* 

to test if torch is installed such that it can access your CUDA installation:

1. Open up the python repl on your command line\
`python`

2. import pytorch\
`import torch`

3. enter this line to check that torch can access your CUDA installation\
`torch.cuda.is_available()`

if this returns as `True` then torch is set up properly to use CUDA. 

if it returns false, consult a guide on installing torch for the right CUDA version. There are many and I've included the following link to one of them for your convenience: 

[how to set up and run cuda operations in pytorch](https://www.geeksforgeeks.org/machine-learning/how-to-set-up-and-run-cuda-operations-in-pytorch/) .

## About

I created this for a friend who has some difficulty with typing out large amounts of text. I've posted it in here in case it might be useful as an accessibility or convenience tool for anyone else. Some of the code to perform specific functions such as entering the text using pyautogui, was written by Gemini and the overall program stitching everything together was written manually. 

The model used for the speech recognition by default is [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3). In my own experience and that of the few friends I've managed to test it with, this model provides incredibly accurate speech recognition output in a fairly short amount of time. (It runs almost in realtime on my computer and I'm using an Nvidia GeForce 2070 Super). If you find the program takes too long to process speech into text, you might try using the turbo model: [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)

The model could easily be changed to another one in the [transformers](https://huggingface.co/docs/transformers/en/index) family simply by changing the text in this line to match whatever model you want to use:

`model_id = "openai/whisper-large-v3"`

## Installation

### install required python modules

`pip install transformers pyautogui keyboard`

### install torch

use the website for torch:

[torch](https://pytorch.org)

installing pytorch for the correct CUDA version will ensure the GPU is available to the whisper model. This isn't required but the program will operate considerably slower without the GPU. 

to select a version compatible with your OS/CUDA version, use the widget on the torch website to get the proper pip3 command and paste the output into your command line.
*torch for CUDA 12.8 is compatible with all CUDA 12.x versions 12.8 and higher* 

to test if torch is installed such that it can access your CUDA installation:

1. Open up the python repl on your command line\
`python`

2. import pytorch\
`import torch`

3. enter this line to check that torch can access your CUDA installation\
`torch.cuda.is_available()`

if this returns as `True` then torch is set up properly to use CUDA. 

if it returns false, consult a guide on installing torch for the right CUDA version. There are many and I've included the following link to one of them for your convenience: 

[how to set up and run cuda operations in pytorch](https://www.geeksforgeeks.org/machine-learning/how-to-set-up-and-run-cuda-operations-in-pytorch/) .