from dotenv import load_dotenv
import os
import io
import logging
from config import *
from multiprocessing import Process, Queue, Event, Value, Lock
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock as LockClass
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass
from multiprocessing.queues import Queue as QueueClass
import threading
import time

from src.streaming.logging_config import setup_logging, start_listener, stop_listener, setup_worker_logging, get_logger, get_log_queue
from src.streaming.voice_recorder import Recorder
from src.streaming.stt_whisper import STTWhisper
from src.streaming.utils import save_wav_file, is_queue_empty
from src.streaming.llm_wrapper import LLMWrapper
from src.streaming.rag_langchain import RAGLangchain
from src.streaming.web_search import WebSearcher
from src.streaming.tts_orpheus import TTSOrpheus
from src.streaming.tts_coqui import TTSCoqui
from src.streaming.tts_kokoro import TTSKokoro
from src.streaming.tts_xtts import TTSXtts
from src.streaming.audio_output import AudioOutputter


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def wake_word_stt_worker(
  interrupt_count : SynchronizedClass,
  voice_setup_event : EventClass,
  pipeline_setup_event: EventClass,
  command_queue: QueueClass,
  command_queue_lock: LockClass,
  log_queue
):
  setup_worker_logging(log_queue)
  logger = get_logger("speech_to_speech.voice_worker")
  logger.debug("Setting up WakeWord and STT")
  
  audio_speaker = AudioOutputter(interrupt_count, logger)
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

      if not is_queue_empty(command_queue_lock, command_queue):
        # If queue is not empty, it means that pipeline is still processing it
        # increment interrupt count
        logger.warning("interrupt fired")
        interrupt_count.value += 1

    logger.debug("Listening for command...")
    command_buffer, command_duration = audio_recorder.record_command(ask_wakeword, command_queue, command_queue_lock, interrupt_count)
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
    # audio_speaker.play_wav_file(command_buffer)

    prev_text = text

    # Add 2 events, 1 to determine when pipeline starts working, and 1 to determine when pipeline stops working on  queue
    command_queue.put({"text": text, "marker": "start", "continuation": continuation, "timestamp": last_command_time})
    command_queue.put({"text": text, "marker": "finish", "continuation": continuation, "timestamp": last_command_time})
    first_loop = False

    audio_buffer_signal.clear()
    audio_buffer_thread.join()


def websearch_llm_tts_worker(
  interrupt_count : SynchronizedClass,
  voice_setup_event : EventClass,
  pipeline_setup_event: EventClass,
  command_queue: QueueClass,
  command_queue_lock: LockClass,
  log_queue
):
  setup_worker_logging(log_queue)
  logger = get_logger("speech_to_speech.pipeline_worker")
  logger.debug("Setting up Websearch, LLM and TTS")
  
  audio_speaker = AudioOutputter(interrupt_count, logger)
  rag = RAGLangchain(interrupt_count=interrupt_count)
  llm = LLMWrapper(
    interrupt_count=interrupt_count,
    memories = rag.get_memories()
  )
  websearch = WebSearcher(interrupt_count=interrupt_count)
  
  if TTS_CHOICE == "coqui":
    tts = TTSCoqui(interrupt_count=interrupt_count)
  elif TTS_CHOICE == "orpheus":
    tts = TTSOrpheus(interrupt_count=interrupt_count)
  elif TTS_CHOICE == "kokoro":
    tts = TTSKokoro(interrupt_count=interrupt_count)
  elif TTS_CHOICE == "xtts":
    tts = TTSXtts(interrupt_count=interrupt_count)
    
  project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  output_dir = os.path.join(project_root_dir, "conversation")
        
  def interrupt_actions(text, info):
    logger.warning(f"Pipeline Interrupted: {info}")
    llm.interrupt_context.append(text)
    end_marker = command_queue.get()
    logger.debug(end_marker)
    interrupt_count.value -= 1
    
    
  def searching_speech_worker(tts, text, interrupt_count, logger):
    audio_speaker = AudioOutputter(interrupt_count, logger)
    output_buffer, output_duration = tts.synthesize(text)
    output_buffer.seek(0)
    audio_speaker.play_wav_file(output_buffer)

  pipeline_setup_event.set()
  logger.debug("Waiting for voice recording to setup")
  voice_setup_event.wait()
  
  while True:
    if not is_queue_empty(command_queue_lock, command_queue):
      work = command_queue.get()
      logger.debug(work)
      text = work["text"]
      marker = work["marker"]
      timestamp = work["timestamp"]
      continuation = work["continuation"]
      
      if marker == "start":
        
        if continuation:
          if len(llm.interrupt_context) > 0:
            llm.interrupt_context.pop()

        # run core pipeline
        if interrupt_count.value > 0:
          interrupt_actions(text, info="before pipeline start")
          continue
        
        # First decide if websearch is needed for prompt
        (decision, topic), memory = llm.decide_websearch_memory(text)
        if interrupt_count.value > 0:
          interrupt_actions(text, info="Decide websearch memory")
          continue
        logger.debug(f"Websearch recommended?: {decision} - {topic}")

        query_results = []

        # IF websearch recommended, check if it is really needed
        if decision == "yes":  
          # First query RAG to see if there exists info in RAG
          logger.debug(f"Querying RAG")
          query_results = rag.query(text)
          if interrupt_count.value > 0:
            interrupt_actions(text, info="querying RAG")
            continue
          # add extra 0 in case RAG is empty and returns empty list
          query_result_scores = [query["score"] for query in query_results] + [0]

          # Only perform websearch if not good data in RAG
          if max(query_result_scores) < RAG_CONFIDENCE_THRESHOLD:
            logger.debug(f"Not enough confident info in RAG, perform search")

            # PLay search speech while web search occurs
            speech_thread = threading.Thread(target=searching_speech_worker, args=(tts, f"Searching the web for {topic}", interrupt_count, logger), daemon=True)
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
              try:
                rag.add_document(document)
              except Exception as e:
                logger.warning(e)
                pass
                
            if interrupt_detected:
              continue

            speech_thread.join()

          else:
            logger.debug(f"Info in RAG exists, no search needed")
        
        # Query the RAG
        logger.debug(f"Querying RAG")
        query_results = rag.query(topic)
        logger.info(query_results)
        if interrupt_count.value > 0:
          interrupt_actions(text, info="Querying RAG")
          continue
            
        context = ""
        # Parse contents from RAG query
        for result in query_results:
          if result["score"] >= RAG_CONFIDENCE_THRESHOLD:
            context += result["content"]

        # Send text and context to LLM for response
        logger.debug("Sending to LLM")
        rag.add_memory(memory, timestamp)
        response = llm.send_to_llm(text, timestamp, memory, context)
        if interrupt_count.value > 0:
          interrupt_actions(text, info="LLM Query")
          continue
        logger.info(response)
        
        # Create speach from response
        logger.debug("Synthesizing speech")
        output_buffer = None
        output_duration = 0
        if TTS_AUDIO_STREAMING:
          output_buffer, output_duration = tts.synthesize_and_stream(response)
        else:
          output_buffer, output_duration = tts.synthesize(response)
        if interrupt_count.value > 0:
          interrupt_actions(text, info="Synthesizing speach for LLM response")
          continue

        output_buffer.seek(0)
        output_filename = os.path.join(output_dir, "output.wav")
        logger.debug("Saving wav file.")
        save_wav_file(output_buffer, response, output_filename, logger)
        output_buffer.seek(0)
        # logger.error(f"End - {time.time()}")
        
        # logger.warning(output_duration)
        
        if not TTS_AUDIO_STREAMING:
          logger.debug("Playing response")
          audio_speaker.play_wav_file(output_buffer)
          if interrupt_count.value > 0:
            interrupt_actions(text, info="Playing Speech")
            continue
          output_buffer.seek(0)

        # Finished without being interrupted, clear llm interrupt_context
        llm.interrupt_context.clear()

        # after finishing work, pop out end marker
        command_queue.get()
        
      # elif marker == "finish":
      #   #only possible if pipeline was interrupted
      #   interrupt_count.value -= 1
  
  


def main():
  listener = start_listener()
  logger = setup_logging()  
  log_queue = get_log_queue()
  
  command_queue = Queue()
  command_queue_lock = Lock()
  interrupt_count = Value("i", 0)
  voice_setup_event = Event()
  pipeline_setup_event = Event()
  # initialize first here to prevent race condition to initialize later
  audio_speaker = AudioOutputter(interrupt_count, logger)
  
  voice_worker = Process(target=wake_word_stt_worker, args=(interrupt_count, voice_setup_event, pipeline_setup_event, command_queue, command_queue_lock, log_queue))
  pipeline_worker = Process(target=websearch_llm_tts_worker, args=(interrupt_count, voice_setup_event, pipeline_setup_event, command_queue, command_queue_lock, log_queue))
  
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