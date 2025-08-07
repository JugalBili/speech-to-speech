import wave
import logging
import io
import numpy as np
import torch
from config import *
from kokoro import KPipeline

logger = logging.getLogger("speech_to_speech.tts_kokoro")

class TTSKokoro:
  def __init__(self):
    torch.backends.cudnn.benchmark = False  # prevents unsupported plan attempts
    torch.backends.cuda.matmul.allow_tf32 = True  # small perf boost on Ampere+
    
    self.client = KPipeline(lang_code=KOKORO_TTS_LANG, device=DEVICE, repo_id='hexgrad/Kokoro-82M') # lang code a = american, b = british
    self.voice = KOKORO_TTS_VOICE
    self.samplerate = 24000
    
  def synthesize(self, text):
    wav_buffer = io.BytesIO()
    wav_file = wave.open(wav_buffer, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(self.samplerate)
    
    audio_duration = 0
    
    try:
      generator = self.client(text, voice=self.voice, speed=1.3 if self.voice == "af_nicole" else 1.0)
      
      for _, _,audio in generator:
        audio = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
        audio = audio.numpy()
        audio *= 32767
        
        audio_duration = len(audio) / 24000 
        wav_Data = audio.astype(np.int16, copy=False)
        wav_file.writeframes(wav_Data.tobytes())

    finally:
      wav_file.close()
    
    return wav_buffer, audio_duration
    
    