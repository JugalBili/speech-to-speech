# from orpheus_tts import OrpheusModel
import wave
import logging
import io
import threading
import queue
import asyncio
from snac import SNAC
from openai import OpenAI
from config import *
import os
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass
from src.streaming.audio_output import AudioOutputter

class TTSOrpheus:
  def __init__(self, interrupt_count: SynchronizedClass):
    self.logger = logging.getLogger("speech_to_speech.tts_orpheus")
    self.interrupt_count = interrupt_count

    self.api = os.getenv("OPENAI_API")
    self.api_key = os.getenv("OPENAI_API_KEY")
    self.model = ORPHEUS_TTS_MODEL
    self.voice = ORPHEUS_TTS_VOICE
    self.snac_device = DEVICE
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
    from src.streaming.tts_orpheus_decoder import convert_to_audio as orpheus_convert_to_audio

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
    thread = threading.Thread(target=run_async, daemon=True)
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
      temperature=ORPHEUS_TTS_TEMPERATURE,
      top_p=ORPHEUS_TTS_TOP_P,
      stream=True,
      # max_tokens=ORPHEUS_TTS_MAX_TOKENS,
      extra_body={ "repeat_penalty": ORPHEUS_TTS_REPEAT_PENALTY }
    )

    # Process the streamed response
    token_counter = 0
    for chunk in response:
      if self.interrupt_count.value > 0:
        return None

      token_counter += 1
      yield chunk.choices[0].text
    
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
      if self.interrupt_count.value > 0:
        return None, None
      audio_segments, audio_duration = self._tokens_decoder_sync(speech_token_generator, wav_file)

    finally:
      wav_file.close()
      
    return wav_buffer, audio_duration

  def synthesize_and_stream(self, text): 
    wav_buffer = io.BytesIO()
    wav_file = wave.open(wav_buffer, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(24000)
    
    audio_duration = 0
    audio_segments = None
    
    try:
      speech_token_generator = self._generate_tokens_from_api(text)
      if self.interrupt_count.value > 0:
        return None, None
      audio_segments, audio_duration = self._tokens_decoder_sync(speech_token_generator, wav_file)

      self.logger.debug("Playing response")
      wav_buffer.seek(0)
      audio_speaker = AudioOutputter(self.interrupt_count, self.logger)
      audio_speaker.play_wav_file(wav_buffer)
      wav_buffer.seek(0)
  
    finally:
      wav_file.close()
      
    return wav_buffer, audio_duration
  