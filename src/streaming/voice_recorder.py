import pvporcupine
from pvrecorder import PvRecorder
from pvspeaker import PvSpeaker
import pvcobra
import numpy as np
import wave
import io
import struct
import logging
import os
from config import *
from multiprocessing import Event
from multiprocessing.synchronize import Event as EventClass
from src.streaming.utils import is_queue_empty


class AudioBuffer:
  def __init__(self, recorder, cobra, framelength, buffer_signal: EventClass):
    self.cobra = cobra
    self.recorder = recorder
    self.buffer_signal = buffer_signal
    self.logger = logging.getLogger("speech_to_speech.voice_recording_buffer")

    self.framelength = framelength
    self.frame_duration = self.framelength / self.recorder.sample_rate
    self.buffer_size = int(AUDIO_BUFFER_DURATION / self.frame_duration)
    
    self.pcm_buffer = [None] * self.buffer_size
    self.pos = 0
    self.voice_frames = 0
    self.full = False 


  def get_buffer(self):
    buffer = self.pcm_buffer[:self.pos]
    if self.full:
      buffer = self.pcm_buffer[self.pos:] + self.pcm_buffer[:self.pos]

    return buffer, self.voice_frames


  def clear_buffer(self):
    self.pcm_buffer = [None] * self.buffer_size
    self.pos = 0
    self.voice_frames = 0
    self.full = False


  def fill_buffer(self):
    silence_threshold_sec = SILENCE_THRESHOLD # Change to 0.5 if you prefer
    silence_frames_required = int(silence_threshold_sec / self.frame_duration)
    silence_frame_count = 0
    voice_frame_count = 0
    total_frame_count = 0

    try:
      # self.logger.warning("Starting Buffer Recording")
      self.recorder.start()

      while self.buffer_signal.is_set():
        frame = None
        try:
          frame = self.recorder.read()
        except Exception as e:
          continue
        pcm = np.array(frame, dtype=np.int16)
        total_frame_count += pcm.size
        
        self.pcm_buffer[self.pos] = frame
        self.pos = (self.pos + 1) % self.buffer_size
        # if we go in circle, set full to be true
        if self.pos == 0:
          self.full = True

        voice_prob = self.cobra.process(pcm)
        # print(voice_prob)

        if voice_prob <= VOICE_PROBABILITY:
          silence_frame_count += 1
        else:
          voice_frame_count += 1
          silence_frame_count = 0

        # print(silence_frame_count)

        if silence_frame_count >= silence_frames_required:
          self.pos = (self.pos - 1) % self.buffer_size
          # reset full flag
          if self.pos == self.buffer_size - 1:
            self.full = False
          self.pcm_buffer[self.pos] = None

    finally:
      self.voice_frames = voice_frame_count
      # self.logger.warning("Stopping Buffer Recording")
      self.recorder.stop()


class Recorder:
  def __init__(self, buffer_signal: EventClass):
    self.logger = logging.getLogger("speech_to_speech.voice_recording")
    self.porcupine = pvporcupine.create(
      access_key=os.getenv("PICOVOICE_API_KEY"),
      keywords=[WAKE_KEYWORD]
    )

    self.cobra = pvcobra.create(
      access_key=os.getenv("PICOVOICE_API_KEY"),
    )
  
    self.framelength =  self.porcupine.frame_length
    self.recorder_device = AUDIO_IN_DEVICE
    self.recorder = PvRecorder(frame_length=self.framelength, device_index=self.recorder_device)
    self.audio_buffer = AudioBuffer(self.recorder, self.cobra, self.framelength, buffer_signal)


  def get_audio_buffer_instance(self):
    return self.audio_buffer
  

  def record_wake_word(self):
    try:
      self.recorder.start()

      while True:
        frame = self.recorder.read()
        pcm = np.array(frame, dtype=np.int16)
        keyword_index = self.porcupine.process(pcm)
        if keyword_index >= 0:
          self.logger.debug("Wake word detected!")
          break   
    finally:
      self.recorder.stop()      
    
  def record_command(self, ask_wakeword=None, command_queue=None, command_queue_lock=None, interrupt_count=None):
    wav_buffer = io.BytesIO()
    wav_file = wave.open(wav_buffer, 'wb')
    wav_file.setparams((1, 2, self.recorder.sample_rate, self.recorder.frame_length, "NONE", "NONE"))

    frame_duration = self.framelength / self.recorder.sample_rate  # Typically 512 / 16000 = 0.032s
    silence_threshold_sec = SILENCE_THRESHOLD # Change to 0.5 if you prefer
    voice_interrupt_threshold_sec = VOICE_THRESHOLD # seconds of total voice required to consider as interrupt
    silence_frames_required = int(silence_threshold_sec / frame_duration)
    # print(silence_frames_required)
    silence_frame_count = 0
    voice_interrupt_frames_required = int(voice_interrupt_threshold_sec / frame_duration)
    voice_frame_count = 0
    interrupt_fired = False
    total_frame_count = 0

    try:
      # write audio buffer contents to file
      frames, voice_frame_count = self.audio_buffer.get_buffer()
      for frame in frames:
        wav_file.writeframes(struct.pack("h" * len(frame), *frame))
      total_frame_count = len(frames)

      self.recorder.start()

      while True:
        frame = None
        try:
          frame = self.recorder.read()
        except Exception as e:
          continue
        pcm = np.array(frame, dtype=np.int16)
        total_frame_count += pcm.size

        wav_file.writeframes(struct.pack("h" * len(frame), *frame))

        voice_prob = self.cobra.process(pcm)
        # print(voice_prob)

        if voice_prob <= VOICE_PROBABILITY:
          silence_frame_count += 1
        else:
          voice_frame_count += 1
          silence_frame_count = 0

        # print(silence_frame_count)

        if silence_frame_count >= silence_frames_required:
          self.logger.debug("Silence detected, stopping recording")
          break

        if voice_frame_count >= voice_interrupt_frames_required:
          if ask_wakeword != None and command_queue and interrupt_count:
            # Ensure only 1 interrupt is fired
            if not ask_wakeword and not is_queue_empty(command_queue_lock, command_queue) and not interrupt_fired:
              self.logger.warning("interrupt fired")
              interrupt_fired = True
              interrupt_count.value += 1

    finally:
      self.recorder.stop()
      wav_file.close()
      interrupt_fired = False

    duration_sec = total_frame_count / self.recorder.sample_rate
    
    return wav_buffer, duration_sec
      
      
      
    
        
        