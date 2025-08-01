from faster_whisper import WhisperModel
import logging

logger = logging.getLogger("speech_to_speech.stt_whisper")

class STTWhisper:
  def __init__(self, vad_active, device):
    self.model_size = "turbo"
    self.vad = vad_active
    self.device = device

    self.whisper_model = WhisperModel(self.model_size, device=self.device, compute_type="int8_float16")
  
  def transcribe(self, audio_buffer):
    segments, _ = self.whisper_model.transcribe(
      audio_buffer,
      language="en",
      vad_filter=self.vad,
      vad_parameters=dict(min_silence_duration_ms=500),
      beam_size=5)
    
    logger.debug("🌐 Speech Parsed!")
    
    return list(segments)
    