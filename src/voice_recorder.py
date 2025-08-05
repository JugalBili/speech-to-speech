import pvporcupine
from pvrecorder import PvRecorder
from pvspeaker import PvSpeaker
import pvcobra
import numpy as np
import wave
import io
import struct
import logging
from config import *

logger = logging.getLogger("speech_to_speech.voice_recording")

class Recorder:
  def __init__(self, PICOVOICE_KEY):
    self.porcupine = pvporcupine.create(
      access_key=PICOVOICE_KEY,
      keywords=[WAKE_KEYWORD]
    )

    self.cobra = pvcobra.create(
      access_key=PICOVOICE_KEY,
    )
  
    self.framelength =  self.porcupine.frame_length
    self.recorder_device = AUDIO_IN_DEVICE
    self.recorder = PvRecorder(frame_length=self.framelength, device_index=self.recorder_device)


  def record_wake_word(self):
    try:
      self.recorder.start()

      while True:
        frame = self.recorder.read()
        pcm = np.array(frame, dtype=np.int16)
        keyword_index = self.porcupine.process(pcm)
        if keyword_index >= 0:
          logger.debug("Wake word detected!")
          break   
    finally:
      self.recorder.stop()      
    
  def record_command(self):
    wav_buffer = io.BytesIO()
    wav_file = wave.open(wav_buffer, 'wb')
    wav_file.setparams((1, 2, self.recorder.sample_rate, self.recorder.frame_length, "NONE", "NONE"))

    frame_duration = self.framelength / self.recorder.sample_rate  # Typically 512 / 16000 = 0.032s
    silence_threshold_sec = SILENCE_THRESHOLD # Change to 0.5 if you prefer
    silence_frames_required = int(silence_threshold_sec / frame_duration)
    print(silence_frames_required)
    silence_frame_count = 0

    try:
      self.recorder.start()

      while True:
        frame = self.recorder.read()
        pcm = np.array(frame, dtype=np.int16)

        wav_file.writeframes(struct.pack("h" * len(frame), *frame))

        voice_prob = self.cobra.process(pcm)
        print(voice_prob)

        if voice_prob <= VOICE_PROBABILITY:
          silence_frame_count += 1
        else:
          silence_frame_count = 0

        print(silence_frame_count)

        if silence_frame_count >= silence_frames_required:
          logger.debug("Silence detected, stopping recording")
          break

    finally:
      self.recorder.stop()
      wav_file.close()

    return wav_buffer
      
      
      
    
        
        