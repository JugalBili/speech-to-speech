from dotenv import load_dotenv
import os
import io
import logging
from config import *
from multiprocessing import Process, Queue, Event, Value
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass
from multiprocessing.queues import Queue as QueueClass
import threading
import time

from src.streaming.logging_config import setup_logging, start_listener, stop_listener, setup_worker_logging, get_logger, get_log_queue
from src.streaming.voice_recorder import Recorder
from src.streaming.stt_whisper import STTWhisper
from src.streaming.utils import save_wav_file, play_wav_file
from src.streaming.llm_wrapper import LLMWrapper
from src.streaming.rag_langchain import RAGLangchain
from src.streaming.web_search import WebSearcher
from src.streaming.tts_orpheus import TTSOrpheus
from src.streaming.tts_coqui import TTSCoqui
from src.streaming.tts_kokoro import TTSKokoro


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def wake_word_stt_worker(
  interrupt_count : SynchronizedClass,
  voice_setup_event : EventClass,
  pipeline_setup_event: EventClass,
  command_queue: QueueClass,
  log_queue
):
  setup_worker_logging(log_queue)
  logger = get_logger("speech_to_speech.voice_worker")
  logger.debug("Setting up WakeWord and STT")
  
  audio_buffer_signal = Event()
  audio_recorder = Recorder(audio_buffer_signal)
  audio_buffer = audio_recorder.get_audio_buffer_instance()
  whisper = STTWhisper(vad_active=True, device=DEVICE)
  project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  output_dir = os.path.join(project_root_dir, "conversation")
  
  first_loop = True
  ask_wakeword = True
  last_command_time = 0
  prev_text = ""

  voice_setup_event.set()
  logger.debug("Waiting for Pipeline to setup")
  pipeline_setup_event.wait()

  while True:
    if (time.time() - last_command_time) > WAKEWORD_RESET_TIME:
      logger.warning("wakeword reset")
      ask_wakeword = True

    if ask_wakeword:
      logger.debug("Listening for wake word...")
      audio_recorder.record_wake_word()
      ask_wakeword = False
      audio_buffer.clear_buffer()

      if not command_queue.empty():
        # If queue is not empty, it means that pipeline is still processing it
        # increment interrupt count
        logger.warning("interrupt fired")
        interrupt_count.value += 1

    logger.debug("Listening for command...")
    command_buffer, command_duration = audio_recorder.record_command(ask_wakeword, command_queue, interrupt_count)
    # logger.error(f"Start - {time.time()}")

    command_buffer.seek(0, io.SEEK_END)
    command_size = command_buffer.tell() # size of command buffer in bytes
    command_buffer.seek(0)
    
    num_samples = command_size // 2
    if num_samples < audio_recorder.porcupine.sample_rate * (VOICE_THRESHOLD + SILENCE_THRESHOLD):
      logger.debug("No speech detected.")
      continue
    
    extra_time_start = time.time()
    audio_buffer.clear_buffer()
    audio_buffer_thread  = threading.Thread(target=audio_buffer.fill_buffer, args=(), daemon=True)
    audio_buffer_signal.set()
    audio_buffer_thread.start()

    logger.debug("Running Speech-To-Text")
    command_buffer.seek(0)
    text_segments = whisper.transcribe(command_buffer)
    text = ", ".join([segment.text for segment in text_segments])
    logger.info(text)

    if not text:
      logger.debug("No command detected")
      audio_buffer_signal.clear()
      audio_buffer_thread.join()
      continue
    
    output_filename = os.path.join(output_dir, "command.wav")
    logger.debug("Saving wav file.")
    command_buffer.seek(0)
    save_wav_file(command_buffer, text, output_filename, logger)

    continuation = False
    extra_time_stop = time.time()
    # if time between this command and the previous one is < threashold, then we treet as continuation of previous prompt
    if WAKEWORD_RESET_TIME > 0 and not first_loop and (time.time() - command_duration - (extra_time_stop - extra_time_start) - last_command_time) < (CONTINUATION_THRESHOLD):
      continuation = True
      text = prev_text + ", " + text

    last_command_time = time.time()

    # command_buffer.seek(0)
    # play_wav_file(command_buffer, logger, interrupt_count)

    prev_text = text

    # Add 2 events, 1 to determine when pipeline starts working, and 1 to determine when pipeline stops working on  queue
    command_queue.put({"text": text, "marker": "start", "continuation": continuation})
    command_queue.put({"text": text, "marker": "finish", "continuation": continuation})
    first_loop = False

    audio_buffer_signal.clear()
    audio_buffer_thread.join()


def websearch_llm_tts_worker(
  interrupt_count : SynchronizedClass,
  voice_setup_event : EventClass,
  pipeline_setup_event: EventClass,
  command_queue: QueueClass,
  log_queue
):
  setup_worker_logging(log_queue)
  logger = get_logger("speech_to_speech.pipeline_worker")
  logger.debug("Setting up Websearch, LLM and TTS")
  
  llm = LLMWrapper(
    interrupt_count=interrupt_count
  )
  websearch = WebSearcher(interrupt_count=interrupt_count)
  rag = RAGLangchain(interrupt_count=interrupt_count)
  
  if TTS_CHOICE == "coqui":
    tts = TTSCoqui(interrupt_count=interrupt_count)
  elif TTS_CHOICE == "orpheus":
    tts = TTSOrpheus(interrupt_count=interrupt_count)
  elif TTS_CHOICE == "kokoro":
    tts = TTSKokoro(interrupt_count=interrupt_count)
    
  project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  output_dir = os.path.join(project_root_dir, "conversation")
        
  def interrupt_actions(text, info):
    logger.warning(f"Pipeline Interrupted: {info}")
    llm.interrupt_context.append(text)
    
  def searching_speech_worker(tts, text, interrupt_count):
    output_buffer, output_duration = tts.synthesize(text)
    output_buffer.seek(0)
    play_wav_file(output_buffer, logger, interrupt_count)

  pipeline_setup_event.set()
  logger.debug("Waiting for voice recording to setup")
  voice_setup_event.wait()
  
  while True:
    if not command_queue.empty():
      work = command_queue.get()
      logger.debug(work)
      text = work["text"]
      marker = work["marker"]
      continuation = work["continuation"]
      
      if marker == "start":
        
        if continuation:
          llm.interrupt_context.pop()

        # run core pipeline
        if interrupt_count.value > 0:
          interrupt_actions(text, info="before pipeline start")
          continue
        
        # First query RAG to see if there exists info in RAG
        query_results = []
        logger.debug(f"Querying RAG")
        query_results = rag.query(text)
        if interrupt_count.value > 0:
          interrupt_actions(text, info="querying RAG")
          continue
        # add extra 0 in case RAG is empty and returns empty list
        query_result_scores = [query["score"] for query in query_results] + [0]
        
        # Only check for websearch if not good data in RAG
        if max(query_result_scores) < RAG_CONFIDENCE_THRESHOLD:
          logger.debug(f"Not enough confident info in RAG, decide search")
          decision, topic = llm.decide_websearch(text)
          if interrupt_count.value > 0:
            interrupt_actions(text, info="Decide websearch")
            continue
          logger.debug(f"Websearch recommended?: {decision} - {topic}")

          # IF websearch needed, perform websearch
          if decision == "yes":           
            # PLay search speech while web search occurs
            speech_thread = threading.Thread(target=searching_speech_worker, args=(tts, f"Searching the web for {topic}", interrupt_count), daemon=True)
            speech_thread.start()
            
            # Get list of websites
            websites = websearch.ddg_search(topic)
            if interrupt_count.value > 0:
              interrupt_actions(text, info="DDG Search")
              continue
            
            # Fetch content from websites
            logger.debug(f"Fetching website contents")
            web_contents = websearch.fetch_content(websites)
            if interrupt_count.value > 0:
              interrupt_actions(text, info="Fetching website contents")
              continue
            
            # Add website data to RAG
            logger.debug(f"Adding contents to RAG")
            interrupt_detected = False
            for document in web_contents:
              logging.info(document)
              if interrupt_count.value > 0:
                interrupt_actions(text, info="Adding document to RAG")
                interrupt_detected = True
                break
              rag.add_document(document)
            if interrupt_detected:
              continue
            
            # Update the RAG query
            logger.debug(f"Querying RAG Again")
            query_results = rag.query(topic)
            logger.info(query_results)
            if interrupt_count.value > 0:
              interrupt_actions(text, info="Querying RAG")
              continue

            speech_thread.join()

        logger.debug(f"Info in RAG exists, no search needed")
        
        context = ""
        # Parse contents from RAG query
        for result in query_results:
          context += result["content"]

        # Send text and context to LLM for response
        logger.debug("Sending to LLM")
        response = llm.send_to_llm(text, context)
        if interrupt_count.value > 0:
          interrupt_actions(text, info="LLM Query")
          continue
        logger.info(response)
        
        # Create speach from response
        logger.debug("Synthesizing speech")
        output_buffer, output_duration = tts.synthesize(response)
        if interrupt_count.value > 0:
          interrupt_actions(text, info="Synthesizing speach for LLM response")
          continue

        output_buffer.seek(0)
        output_filename = os.path.join(output_dir, "output.wav")
        logger.debug("Saving wav file.")
        save_wav_file(output_buffer, response, output_filename, logger)
        output_buffer.seek(0)
        logger.error(f"End - {time.time()}")
        logger.debug("Playing response")
        play_wav_file(output_buffer, logger, interrupt_count=interrupt_count)
        if interrupt_count.value > 0:
          interrupt_actions(text, info="Playing Speech")
          continue
        output_buffer.seek(0)

        # Finished without being interrupted, clear llm interrupt_context
        llm.interrupt_context.clear()

        # after finishing work, pop out end marker
        command_queue.get()
        
      elif marker == "finish":
        #only possible if pipeline was interrupted
        interrupt_count.value -= 1
  
  


def main():
  listener = start_listener()
  logger = setup_logging()  
  log_queue = get_log_queue()
  
  command_queue = Queue()
  interrupt_count = Value("i", 0)
  voice_setup_event = Event()
  pipeline_setup_event = Event()
  
  voice_worker = Process(target=wake_word_stt_worker, args=(interrupt_count, voice_setup_event, pipeline_setup_event, command_queue, log_queue))
  pipeline_worker = Process(target=websearch_llm_tts_worker, args=(interrupt_count, voice_setup_event, pipeline_setup_event, command_queue, log_queue))
  
  logger.debug("Starting Processes")
  voice_worker.start()
  pipeline_worker.start()
  
  voice_worker.join()
  pipeline_worker.join()
  logger.debug("Processes Completed")
  
    
  stop_listener()

if __name__ == "__main__":
  load_dotenv()
  main()