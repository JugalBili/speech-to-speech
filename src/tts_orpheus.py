# from orpheus_tts import OrpheusModel
import wave
import logging
import io
import os
import sys
import requests
import json
import time
import numpy as np
import sounddevice as sd
import threading
import queue
import asyncio
from snac import SNAC
from openai import OpenAI

logger = logging.getLogger("speech_to_speech.tts_orpheus")

# class TTSOrpheus:
#   def __init__(self):
#     self.model = "canopylabs/orpheus-tts-0.1-finetune-prod"
#     self.client = OrpheusModel(model_name=self.model)
#     self.voice = "tara"
    
#   def synthesize(self, text):
#     wav_buffer = io.BytesIO()
#     wav_file = wave.open(wav_buffer, 'wb')
#     wav_file.setnchannels(1)
#     wav_file.setsampwidth(2)
#     wav_file.setframerate(24000)
    
#     audio_duration = 0
  
#     try:
#       tokens = self.client.generate_speech(
#         prompt=text,
#         voice=self.voice,
#         max_buffer_size=5000,
#         top_p=0.95,
#         temperature=0.7,
#         repetition_penalty=1.3,
#         # max_tokens=150
#       )
      
#       total_frames = 0
#       chunk_counter = 0
#       for audio_chunk in tokens: # output streaming
#         chunk_counter += 1
#         frame_count = len(audio_chunk) // (wav_file.getsampwidth() * wav_file.getnchannels())
#         total_frames += frame_count
#         wav_file.writeframes(audio_chunk)
#       duration = total_frames / wav_file.getframerate()
      
#       logger.debug("🎤 Speech synthesized!")

#     finally:
#       wav_file.close()

#     return wav_buffer, audio_duration


class TTSOrpheus:
  def __init__(self, api, api_key, model):
    self.api = api
    self.api_key = api_key
    self.model = model # "orpheus-3b-0.1-ft" # "canopylabs/orpheus-tts-0.1-finetune-prod"
    # self.client = OrpheusModel(model_name=self.model)
    self.voice = "Leo"
    self.snac_device = "cuda"
    self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.snac_device)

    self.START_TOKEN_ID = 128259
    self.END_TOKEN_IDS = [128009, 128260, 128261, 128257]
    self.CUSTOM_TOKEN_PREFIX = "<custom_token_"
    self.SAMPLE_RATE = 24000  # SNAC model uses 24kHz
    
    self.client = OpenAI(base_url=self.api, api_key=self.api_key)


  def _format_prompt(self, text):
    """Format prompt for Orpheus model with voice prefix and special tokens."""
        
    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{self.voice}: {text}"
    
    # Add special token markers for the LM Studio API
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"   # Using the eos_token from config
    
    return f"{special_start}{formatted_prompt}{special_end}"
  

  def _turn_token_into_id(self, token_string, index):
    """Convert token string to numeric ID for audio processing."""
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind(self.CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
      return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith(self.CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
      try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)
        return token_id
      except ValueError:
        return None
    else:
        return None


  def _convert_to_audio(self, multiframe, count):
    """Convert token frames to audio."""
    # Import here to avoid circular imports
    from tts_orpheus_decoder import convert_to_audio as orpheus_convert_to_audio

    return orpheus_convert_to_audio(self.snac_model, self.snac_device, multiframe, count)


  async def _tokens_decoder(self, token_gen):
    """Asynchronous token decoder that converts token stream to audio stream."""
    buffer = []
    count = 0
    async for token_text in token_gen:
      token = self._turn_token_into_id(token_text, count)
      if token is not None and token > 0:
        buffer.append(token)
        count += 1
        
        # Convert to audio when we have enough tokens
        if count % 7 == 0 and count > 27:
          buffer_to_proc = buffer[-28:]
          audio_samples = self._convert_to_audio(buffer_to_proc, count)
          if audio_samples is not None:
            yield audio_samples

  def _tokens_decoder_sync(self, syn_token_gen, wav_file):
    """Synchronous wrapper for the asynchronous token decoder."""
    audio_queue = queue.Queue()
    audio_segments = []
       
    # Convert the synchronous token generator into an async generator
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in self._tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel to indicate completion

    def run_async():
        asyncio.run(async_producer())

    # Start the async producer in a separate thread
    thread = threading.Thread(target=run_async)
    thread.start()

    # Process audio as it becomes available
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        
        audio_segments.append(audio)
        
        # Write to WAV file if provided
        if wav_file:
            wav_file.writeframes(audio)
     
    thread.join()
    
    # Calculate and print duration
    duration = sum([len(segment) // (2 * 1) for segment in audio_segments]) / self.SAMPLE_RATE
    print(f"Generated {len(audio_segments)} audio segments")
    print(f"Generated {duration:.2f} seconds of audio")
    
    return audio_segments, duration
    

  def _generate_tokens_from_api(self, text):
    """Generate tokens from text using LM Studio API."""
    formatted_prompt = self._format_prompt(text)
    print(f"Generating speech for: {formatted_prompt}")
    
    
    response = self.client.completions.create(
      model=self.model,
      prompt=formatted_prompt,
      temperature=0.5,
      top_p=0.9,
      stream=True,
      max_tokens=2048,
      extra_body={ "repeat_penalty": 1.1 }
    )

    # Process the streamed response
    token_counter = 0
    for chunk in response:
      token_counter += 1
      yield chunk.choices[0].text
      
    # token_counter = 0
    # for chunk in response:
    #   line = line.decode('utf-8')
    #   if line.startswith('data: '):
    #       data_str = line[6:]  # Remove the 'data: ' prefix
    #       if data_str.strip() == '[DONE]':
    #           break
              
    #       try:
    #           data = json.loads(data_str)
    #           if 'choices' in data and len(data['choices']) > 0:
    #               token_text = data['choices'][0].get('text', '')
    #               token_counter += 1
    #               if token_text:
    #                   yield token_text
    #       except json.JSONDecodeError as e:
    #           print(f"Error decoding JSON: {e}")
    #           continue
    
    logging.debug("Token generation complete")
    
  def synthesize(self, text): 
    wav_buffer = io.BytesIO()
    wav_file = wave.open(wav_buffer, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(24000)
    
    audio_duration = 0
    audio_segments = None
    
    try:
      speech_token_generator = self._generate_tokens_from_api(text)
      audio_segments, audio_duration = self._tokens_decoder_sync(speech_token_generator, wav_file)
      
    finally:
      wav_file.close()
      
    return wav_buffer, audio_duration
    
  
    # try:
    #   tokens = self.client.generate_speech(
    #     prompt=text,
    #     voice=self.voice,
    #     max_buffer_size=5000,
    #     top_p=0.95,
    #     temperature=0.7,
    #     repetition_penalty=1.3,
    #     # max_tokens=150
    #   )
      
    #   total_frames = 0
    #   chunk_counter = 0
    #   for audio_chunk in tokens: # output streaming
    #     chunk_counter += 1
    #     frame_count = len(audio_chunk) // (wav_file.getsampwidth() * wav_file.getnchannels())
    #     total_frames += frame_count
    #     wav_file.writeframes(audio_chunk)
    #   duration = total_frames / wav_file.getframerate()
      
    #   logger.debug("🎤 Speech synthesized!")

    # finally:
    #   wav_file.close()

    # return wav_buffer, audio_duration
  