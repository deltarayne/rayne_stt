
## rayne_stt

I created this for a friend who has difficulty typing. I've posted it in here in case it might
be useful as an accessibility or convenience tool for anyone else. Some of the code to perform specific functions such as entering the text using pyautogui, was written by Gemini and the overall program stitching everything together was written manually.\ 

The model used for the speech recognition by default is [whisper-large-v3-turbo] (https://huggingface.co/openai/whisper-large-v3-turbo)

The model could easily be change to another one in the [transformers] (https://huggingface.co/docs/transformers/en/index) family simply by changing the text in this line to match whatever model you want to use:\

`model_id = "openai/whisper-large-v3"'

## installation

### install required python modules

`pip install transformers pyautogui keyboard`

### install torch

use the main website for torch:

[pytorch] (https://pytorch.org)

installing pytorch for the correct CUDA version will insure the GPU is available to the whisper model. This isn't required but the program will operate considerably slower without the GPU. 

to select a version compatible with your OS/CUDA version, use the widget on the torch website to get the proper pip3 command and paste the output into your commandline
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

[how to set up and run cuda operations in pytorch] (https://www.geeksforgeeks.org/machine-learning/how-to-set-up-and-run-cuda-operations-in-pytorch/)