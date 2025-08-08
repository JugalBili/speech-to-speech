import wave
import logging
import io
import numpy as np
from TTS.api import TTS
import torch
from config import *
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass

class TTSCoqui:
  def __init__(self, interrupt_count: SynchronizedClass):
    torch.backends.cudnn.benchmark = False  # prevents unsupported plan attempts
    torch.backends.cuda.matmul.allow_tf32 = True  # small perf boost on Ampere+

    self.logger = logging.getLogger("speech_to_speech.tts_xtts")
    self.interrupt_count = interrupt_count
    
    self.model = COQUI_TTS_MODEL
    self.client = TTS(self.model).to(DEVICE)
    self.reference_audio = COQUI_TTS_REFERENCE_WAV
  
  def synthesize(self, text):
    wav_buffer = io.BytesIO()
    wav_file = wave.open(wav_buffer, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(24000)
    
    audio_duration = 0
  
    try:
      wav_values = self.client.tts(
        text=text,
        language="en" if "xtts" in self.model else None,
        speaker_wav=self.reference_audio if "xtts" in self.model else None, 
        speaker=COQUI_TTS_SPEAKER if "xtts" not in self.model else None
      )

      if self.interrupt_count.value > 0:
        return None, None
  
      audio_duration = len(wav_values) / 24000 
      wav_values = np.array(wav_values, dtype=np.float32)
      wav_values *= 32767
      wav_int16 = wav_values.astype(np.int16, copy=False)
      wav_file.writeframes(wav_int16.tobytes())
    
    finally:
      wav_file.close()
    
    return wav_buffer, audio_duration