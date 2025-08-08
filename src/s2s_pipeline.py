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
from src.web_search import WebSearcher
from src.rag_langchain import RAGLangchain
from config import *
import threading

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = logging.getLogger("speech_to_speech.s2s_pipeline")


def searching_speech_worker(tts, text):
  output_buffer, output_duration = tts.synthesize(text)
  output_buffer.seek(0)
  play_wav_file(output_buffer)


def main():
  audio_recorder = Recorder()
  whisper = STTWhisper(vad_active=True, device=DEVICE)
  llm = LLMWrapper()
  websearch = WebSearcher()
  rag = RAGLangchain()
  
  if TTS_CHOICE == "coqui":
    tts = TTSCoqui()
  elif TTS_CHOICE == "orpheus":
    tts = TTSOrpheus()
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
    text = "\n".join([segment.text for segment in text_segments])
    logger.info(text)
    
    # text = "test"  

    decision, topic = llm.decide_websearch(text)
    logger.debug(f"Websearch recommended?: {decision} - {topic}")
    context = ""
    query_results = []

    if decision == "yes":
      logger.debug(f"Querying RAG")
      query_results = rag.query(topic)
      # add extra 0 in case RAG is empty and returns empty list
      query_result_scores = [query["score"] for query in query_results] + [0]

      if  max(query_result_scores) < RAG_CONFIDENCE_THRESHOLD:
        logger.debug(f"Not enough confident info in RAG, requires search")
        
        # PLay search speech while web search occurs
        speech_thread = threading.Thread(target=searching_speech_worker, args=(tts, f"Searching the web for {topic}"))
        speech_thread.start()
        
        websites = websearch.ddg_search(text)
        # print(websites)
        logger.debug(f"Fetching website contents")
        web_contents = websearch.fetch_content(websites)

        # print(web_contents)

        logger.debug(f"Adding contents to RAG")
        for document in web_contents:
          rag.add_document(document)

        logger.debug(f"Querying RAG Again")
        query_results = rag.query(topic)
        logger.info(query_results)

        speech_thread.join()
  
    logger.debug(f"Info in RAG exists, no search needed")
    
    for result in query_results:
      context += result["content"]

    logger.debug("Sending to LLM")
    response = llm.send_to_llm(text, context)
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