import wave
import logging
import io
import numpy as np
import torch
from config import *
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
from src.streaming.audio_output import AudioOutputter

class TTSXtts:
  def __init__(self, interrupt_count: SynchronizedClass):
    torch.backends.cudnn.benchmark = False  # prevents unsupported plan attempts
    torch.backends.cuda.matmul.allow_tf32 = True  # small perf boost on Ampere+

    self.logger = logging.getLogger("speech_to_speech.tts_xtts_v2")
    self.interrupt_count = interrupt_count
    
    self.model_name = XTTS_HUGGINGFACE_MODEL
    checkpoint_path = snapshot_download(self.model_name)
    config_path = f"{checkpoint_path}/config.json"
    self.config = XttsConfig()
    self.config.load_json(config_path )
    self.model = Xtts.init_from_config(self.config)
    self.model.load_checkpoint(self.config, checkpoint_dir=checkpoint_path)
    self.model.cuda()
    self.reference_audio = XTTS_REFERENCE_WAV 
    self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=[self.reference_audio])
    self.sample_rate = 24000
  
  
  def synthesize(self, text):
    wav_buffer = io.BytesIO()
    wav_file = wave.open(wav_buffer, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(self.sample_rate)
    
    audio_duration = 0
  
    try:
      chunks = self.model.inference_stream(
        text,
        "en",
        self.gpt_cond_latent,
        self.speaker_embedding
      )

      wav_chunks = []
      for i, chunk in enumerate(chunks):
        if self.interrupt_count.value > 0:
          break
        wav_chunks.append(chunk)
        # wav = torch.cat(wav_chuncks, dim=0)

      
      wav_values = torch.cat(wav_chunks, dim=0).squeeze().cpu().numpy()

      audio_duration = len(wav_values) / self.sample_rate 
      wav_int16 = np.int16(wav_values * 32767)
      wav_file.writeframes(wav_int16.tobytes())
    
    finally:
      wav_file.close()
    
    return wav_buffer, audio_duration
  
  
  def synthesize_and_stream(self, text):
    wav_buffer = io.BytesIO()
    wav_file = wave.open(wav_buffer, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(24000)
    
    audio_speaker = AudioOutputter(self.interrupt_count, self.logger)
    audio_speaker.start_audio_stream()
    audio_duration = 0

    self.logger.debug("Streaming response")
    try:
      chunks = self.model.inference_stream(
        text,
        "en",
        self.gpt_cond_latent,
        self.speaker_embedding,
        enable_text_splitting=True
      )

      wav_chunks = []
      for i, chunk in enumerate(chunks):
          if self.interrupt_count.value > 0:
            break

          wav_chunks.append(chunk)
          
          wav_values = chunk.squeeze().cpu().numpy()
          wav_pcm = np.int16(wav_values * 32767)

          audio_speaker.play_stream_audio(wav_pcm)

      wav_values = torch.cat(wav_chunks, dim=0).squeeze().cpu().numpy()

      audio_duration = len(wav_values) / self.sample_rate 
      wav_int16 = np.int16(wav_values * 32767)
      wav_file.writeframes(wav_int16.tobytes())
      
      audio_speaker.stop_streaming()
    
    finally:
      wav_file.close()
    
    return wav_buffer, audio_duration