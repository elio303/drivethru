# Instructions

## Overview
This project attempts to mimic a cashier or drive-thru attendant. It does so by using an LLM to understand and respond to customers. It also does audio processing by using various libraries to interpret customer voices into text and transforming the LLM's responses into audio.

## Install global dependencies
```
xcode-select --install
brew install python3 poetry portaudio ffmpeg
```
* pyaudio requires portaudio
* pydub requires ffmpeg

## Install local dependencies
```
cd {WORKING_DIRECTORY}
python3 -m venv .venv
poetry shell
poetry install
```

## Install and Setup LLM Studio
1. Download LLM Studio from https://lmstudio.ai/
2. Install and open LLM Studio
3. On the left menu bar, click the Home icon
4. In LLM Studio's search bar, search for "Phi-3-mini-4k-instruct-GGUF"
5. Download the model
6. On the left menu bar, click the Home icon
7. In the top left component, click on the "Local Server" button
8. On the top middle, select "phi3" from the "Select a model to load" dropdown and wait for the model to load
9. In the top left component, click on the "Start Server" button

## Create your own environment file
```
cp example.env dev.env
```
* Ensure that you fill your environment variables with values in `dev.env`

## Run the application
```
cd {WORKING_DIRECTORY}
python3 main.py env-path=dev.env
```