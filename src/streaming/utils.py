import logging
import wave
import array
import threading
from pvspeaker import PvSpeaker
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass
from multiprocessing.synchronize import Lock as LockClass
from multiprocessing.queues import Queue as QueueClass
from typing import Union
from config import *

def save_wav_file(wav_bytes, text, wav_filename, logger):
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

  with open(wav_filename.replace(".wav", ".txt"), "w") as txt_file:
    txt_file.write(text.replace("\n", " "))

  logger.debug(f"Audio successfully saved to {wav_filename}")


def is_queue_empty(lock: LockClass, queue: QueueClass):
  empty = False

  lock.acquire()
  empty = queue.empty()
  lock.release()

  return empty
