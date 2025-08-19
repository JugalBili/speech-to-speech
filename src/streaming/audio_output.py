import logging
import wave
import array
import threading
from pvspeaker import PvSpeaker
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass
from config import *
import queue
import time


class AudioOutputter():
  _instance = None

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
        cls._instance = super().__new__(cls)
    return cls._instance
   
  def __init__(self, interrupt_count: SynchronizedClass, logger: logging.Logger):
    self.interrupt_count = interrupt_count
    self.logger = logger

    self.speaker = None
    self.monitor_thread = threading.Thread(target=self._interrupt_monitor, daemon=True)
    self._stop_monitor = threading.Event()
    self.monitor_thread.start()
    
    self._stream_queue = queue.Queue()  # buffer ~50 chunks
    self._stream_stop_event = threading.Event()
    self._stream_thread = None


  def _interrupt_monitor(self):
    prev_interrupt_count = 0
    while not self._stop_monitor.is_set():
      count = self.interrupt_count.value
      if count > 0 and count != prev_interrupt_count:
        if self._stream_worker:
          self._stream_stop_event.set()
        if self.speaker:
          self.logger.warning("Speaker interrupted")
          try:
            self.speaker.stop()
          except Exception as e:
            self.logger.error(e)
      prev_interrupt_count = count
      self._stop_monitor.wait(0.1)  # check ~10 times per second
   
    
  def start_audio_stream(self, sample_rate: int = 24000, bits_per_sample: int = 16):
    """Initialize PvSpeaker for streaming."""
    if self.speaker:
        self.speaker.delete()
  
    self.speaker = PvSpeaker(
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        buffer_size_secs=20,
        device_index=AUDIO_OUT_DEVICE
    )
    
    self._stream_stop_event.clear()
    self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
    self.speaker.start()
    self._stream_thread.start()
    
  
  def _stream_worker(self):
    while not self._stream_stop_event.is_set():
        try:
          pcm_chunk = self._stream_queue.get(timeout=0.1)  # wait for new chunk
          total_written = 0
          while total_written < len(pcm_chunk):
            if self.interrupt_count.value > 0:
              self.speaker.stop()
              break
            
            # self.logger.error(f"Chunk written - {time.time()}")
            written = self.speaker.write(pcm_chunk[total_written:])
            total_written += written
        except queue.Empty:
            continue  # no chunk yet, loop again
    
    # go until queue is empty after stop event is given
    while not self._stream_queue.empty():
      pcm_chunk = self._stream_queue.get(timeout=0.1)  # wait for new chunk
      total_written = 0
      while total_written < len(pcm_chunk):
        if self.interrupt_count.value > 0:
          self.speaker.stop()
          break

        written = self.speaker.write(pcm_chunk[total_written:])
        total_written += written


  def play_stream_audio(self, pcm_chunk):
      # Convert numpy int16 to list for PvSpeaker
      pcm_list = pcm_chunk.tolist()
      self._stream_queue.put(pcm_list)


  def stop_streaming(self):
    self.logger.debug("Waiting for audio to finish...")
    self._stream_stop_event.set()
    self._stream_thread.join()
    if self.speaker:
      try:
        self.speaker.flush()
      except Exception as e:
        pass
      self.speaker.stop()
      self.speaker.delete()
      self.speaker = None


  def _split_list(self, input_list, x):
    return [input_list[i:i + x] for i in range(0, len(input_list), x)]


  def play_wav_file(self, wav_bytes):
    wav_file = wave.open(wav_bytes, 'rb')
    sample_rate = wav_file.getframerate()
    bits_per_sample = wav_file.getsampwidth() * 8
    num_channels = wav_file.getnchannels()
    num_samples = wav_file.getnframes()
    
    if bits_per_sample != 8 and bits_per_sample != 16 and bits_per_sample != 24 and bits_per_sample != 32:
      self.logger.error(f"Unsupported bits per sample: {bits_per_sample}")
      wav_file.close()
      exit()

    if num_channels != 1:
      self.logger.error("WAV file must have a single channel (MONO)")
      wav_file.close()
      exit()

    if self.speaker:
      self.speaker.delete()

    self.speaker = PvSpeaker(
      sample_rate=sample_rate,
      bits_per_sample=bits_per_sample,
      buffer_size_secs=20,
      device_index=AUDIO_OUT_DEVICE
    )
    # print("pvspeaker version: %s" % self.speaker.version)
    # print("Using device: %s" % self.speaker.selected_device)

    wav_bytes = wav_file.readframes(num_samples)

    pcm = None
    if bits_per_sample == 8:
        pcm = list(array.array('B', wav_bytes))
    elif bits_per_sample == 16:
        pcm = list(array.array('h', wav_bytes))
    elif bits_per_sample == 24:
        pcm = []
        for i in range(0, len(wav_bytes), 3):
            sample = int.from_bytes(wav_bytes[i:i + 3], byteorder='little', signed=True)
            pcm.append(sample)
    elif bits_per_sample == 32:
        pcm = list(array.array('i', wav_bytes))

    pcm_list = self._split_list(pcm, sample_rate)
    self.speaker.start()

    try:
      # print("Playing audio...")
      for pcm_sublist in pcm_list:
          sublist_length = len(pcm_sublist)
          total_written_length = 0
          while total_written_length < sublist_length:
            if self.interrupt_count and self.interrupt_count.value > 0:
              self.logger.warning("Speaker interrupted during write")
              return

            written_length = self.speaker.write(pcm_sublist[total_written_length:])
            total_written_length += written_length

      self.logger.debug("Waiting for audio to finish...")
      self.speaker.flush()

    finally:
      self.speaker.stop()
      self.speaker.delete()
      self.speaker = None

    self.logger.debug("Finished playing audio...")
    wav_file.close()


  def shutdown(self):
    self._stop_monitor.set()
    self.monitor_thread.join()

    if self.speaker:
      self.speaker.stop()
      self.speaker.delete()
      self.speaker = None