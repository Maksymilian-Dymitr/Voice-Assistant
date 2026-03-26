import os
import queue
import threading
import time

import keyboard
import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from torch.jit import Error
from transformers import AutoModelForCausalLM, AutoTokenizer

gpu = "cuda"
computation = "float16"

whisper = WhisperModel("large-v2", devcie=gpu, compute_type=computation)

sound = pyaudio.PyAudio()
stream = sound.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

frames = []
is_recording = False

def start_recording(): 
    global is_recording
    is_recording = True

def stop_recording(): 
    global is_recording
    is_recording = False
    
def record():
    global is_recording
    try:
       if is_recording:
           print("Recording")
           while is_recording: 
               data = stream.read(1024)
               frames.append(data)
       
       elif not is_recording:
            if not frames:
                return
            audio_data = b"".join(frames)
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            segments, info = whisper.transcribe(audio, beam_size=5)
            for segment in segments:
                print(f"{segment.text}")
    except Error:
        pass


stream.stop_stream() 
stream.close() 
sound.terminate() 

record()
keyboard.add_hotkey("space", start_recording)        
keyboard.add_hotkey("space", stop_recording, trigger_on_release=True)        
keyboard.wait()

