import logging
import wave
import array
import threading
from pvspeaker import PvSpeaker
from config import *

logger = logging.getLogger("speech_to_speech.s2s_pipeline")


def save_wav_file(wav_bytes, wav_filename):
  with wave.open(wav_bytes, 'rb') as in_wav:
    # Get parameters from the input wave file
    nchannels = in_wav.getnchannels()
    sampwidth = in_wav.getsampwidth()
    framerate = in_wav.getframerate()
    nframes = in_wav.getnframes()
    comptype = in_wav.getcomptype()
    compname = in_wav.getcompname()

    # Read all frames
    frames = in_wav.readframes(nframes)

  # Open a new WAV file for writing
  with wave.open(wav_filename, 'wb') as out_wav:
    # Set parameters for the output wave file
    out_wav.setnchannels(nchannels)
    out_wav.setsampwidth(sampwidth)
    out_wav.setframerate(framerate)
    out_wav.setcomptype(comptype, compname)

    # Write the frames to the output file
    out_wav.writeframes(frames)

  logger.debug(f"Audio successfully saved to {wav_filename}")
  
def play_wav_file(wav_bytes):
  
  def blocking_call(speaker):
    speaker.flush()

  def worker_function(speaker, completion_event):
    blocking_call(speaker)
    completion_event.set()

  def split_list(input_list, x):
    return [input_list[i:i + x] for i in range(0, len(input_list), x)]

  wav_file = wave.open(wav_bytes, 'rb')
  sample_rate = wav_file.getframerate()
  bits_per_sample = wav_file.getsampwidth() * 8
  num_channels = wav_file.getnchannels()
  num_samples = wav_file.getnframes()
  
  if bits_per_sample != 8 and bits_per_sample != 16 and bits_per_sample != 24 and bits_per_sample != 32:
    logger.error(f"Unsupported bits per sample: {bits_per_sample}")
    wav_file.close()
    exit()

  if num_channels != 1:
    logger.error("WAV file must have a single channel (MONO)")
    wav_file.close()
    exit()

  speaker = PvSpeaker(
    sample_rate=sample_rate,
    bits_per_sample=bits_per_sample,
    buffer_size_secs=20,
    device_index=AUDIO_OUT_DEVICE
  )
  print("pvspeaker version: %s" % speaker.version)
  print("Using device: %s" % speaker.selected_device)

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

  pcm_list = split_list(pcm, sample_rate)
  speaker.start()

  print("Playing audio...")
  for pcm_sublist in pcm_list:
      sublist_length = len(pcm_sublist)
      total_written_length = 0
      while total_written_length < sublist_length:
          written_length = speaker.write(pcm_sublist[total_written_length:])
          total_written_length += written_length

  logger.debug("Waiting for audio to finish...")

  completion_event = threading.Event()
  worker_thread = threading.Thread(target=worker_function, args=(speaker, completion_event))
  worker_thread.start()
  completion_event.wait()
  worker_thread.join()

  speaker.stop()

  logger.debug("Finished playing audio...")
  wav_file.close()