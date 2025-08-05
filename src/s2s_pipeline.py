import argparse
from dotenv import load_dotenv
import os
import io
import logging

from src.logging_config import setup_logging
setup_logging()

from src.voice_recorder import Recorder
from src.stt_whisper import STTWhisper
from src.llm_wrapper import LLMWrapper
from src.tts_orpheus import TTSOrpheus
from src.tts_coqui import TTSCoqui
from src.tts_kokoro import TTSKokoro
from src.utils import save_wav_file, play_wav_file
from config import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = logging.getLogger("speech_to_speech.s2s_pipeline")


def main():
  audio_recorder = Recorder(os.getenv("PICOVOICE_API_KEY"))
  whisper = STTWhisper(vad_active=True, device=DEVICE)
  llm = LLMWrapper(
    api=os.getenv("OPENAI_API"),
    api_key=os.getenv("OPENAI_API_KEY"),
  )
  
  if TTS_CHOICE == "coqui":
    tts = TTSCoqui()
  elif TTS_CHOICE == "orpheus":
    tts = TTSOrpheus(
      api=os.getenv("OPENAI_API"),
      api_key=os.getenv("OPENAI_API_KEY"),
    )
  elif TTS_CHOICE == "kokoro":
    tts = TTSKokoro()
    
  while True:
    logger.debug("Listening for wake word...")
    audio_recorder.record_wake_word()
    
    
    logger.debug("Listening for command...")
    command_buffer = audio_recorder.record_command()
    
    command_buffer.seek(0, io.SEEK_END)
    command_size = command_buffer.tell() # size of command buffer in bytes
    command_buffer.seek(0)
    
    num_samples = command_size // 2
    if num_samples < audio_recorder.porcupine.sample_rate * 2:
        logger.debug("No speech detected.")
        continue
    
    output_filename = "command.wav"
    logger.debug("Saving wav file.")
    save_wav_file(command_buffer, output_filename)
    

    logger.debug("Running Speech-To-Text")
    command_buffer.seek(0)
    text_segments = whisper.transcribe(command_buffer)
    text = "/n".join([segment.text for segment in text_segments])
    logger.info(text)
    
    # text = "Hello, how are you doing today?"
    
    
    logger.debug("Sending to LLM")
    response = llm.send_to_llm(text)
    logger.info(response)
    
    
    logger.debug("Synthesizing speech")
    output_buffer, output_duration = tts.synthesize(response)

    output_buffer.seek(0)
    output_filename = "output.wav"
    logger.debug("Saving wav file.")
    save_wav_file(output_buffer, output_filename)
    output_buffer.seek(0)
    
    logger.debug("Playing response")
    play_wav_file(output_buffer)
    output_buffer.seek(0)
    
    

if __name__ == "__main__":
  load_dotenv()
  main()