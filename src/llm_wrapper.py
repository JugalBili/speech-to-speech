from openai import OpenAI
import logging
import json
import re
from config import *

logger = logging.getLogger("speech_to_speech.llm_wrapper")

class LLMWrapper():
  def __init__(self, api, api_key):
    self.api = api
    self.api_key = api_key
    self.model = LLM_MODEL
    self.global_chat_history = []
    self.current_chat_history = []
    self.current_chat_history_length = 0
    self.chat_history_filename = "chat_history.json"
    self.max_tokens = int(MAX_TOKENS * 0.75)
    
    self.initial_prompt = INITIAL_PROMPT
    self.initial_prompt += "Do not style your response using markdown formatting. For example, do not encapuslate words with asterisks or "
    if TTS_CHOICE == "orpheus":
      self.initial_prompt += " Also, add paralinguistic elements like <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp> or uhm for more human-like speech whenever it fits, but do not overdo it, please only add it when necessary and not often."
    self.initial_prompt = self.initial_prompt.replace("\n", "")
    self.initial_prompt_length = len(self.initial_prompt.split(" "))
    
    self.client = OpenAI(base_url=self.api, api_key=self.api_key)
    
    self._load_convo_history()
    
    
  def _load_convo_history(self):
    logging.debug("📝 Loading conversation history")

    chat_history_file = open(self.chat_history_filename, 'r', encoding="utf-8")
    self.global_chat_history = json.load(chat_history_file)["history"]
    chat_history_file.close()
    
    index = len(self.global_chat_history) - 1
    while index > -1 and ((self.current_chat_history_length + self.initial_prompt_length) < self.max_tokens ):
      cur_message = self.global_chat_history[index]
      self.current_chat_history.insert(0, cur_message["message"])
      self.current_chat_history_length += cur_message["length"]
      index -= 1
      
    print(self.current_chat_history)


  def _write_chat_history(self):
    logger.debug(f"💾 Saving conversation history")

    chat_history_file = open(self.chat_history_filename, 'w', encoding="utf-8")
    json.dump({
        "history" : self.global_chat_history
      },
      chat_history_file
    )
    chat_history_file.close()


  def _filter_think(self, text):
    marker = "</think>"
    index = text.find(marker) + len(marker)
    filtered_text = text[index:].replace("\n", "")
    
    return filtered_text
  
  def _filter_emoji(self, text):
    emoji_pattern = re.compile("["
      u"\U0001F600-\U0001F64F"  # emoticons
      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
      u"\U0001F680-\U0001F6FF"  # transport & map symbols
      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
      u"\U00002702-\U000027B0"
      u"\U000024C2-\U0001F251"
      "]+", flags=re.UNICODE
    )

    filtered_text = emoji_pattern.sub(r'', text)
    
    return filtered_text


  def send_to_llm(self, text):
    if not ENABLE_THINK:
      text = text + " /no_think" # disable reasoning

    text_length = len(text.split(" "))
    
    self.current_chat_history.append(
      {"role": "user", "content": text}
    )
    
    self.current_chat_history_length += text_length
    
    while self.current_chat_history and ((self.current_chat_history_length + self.initial_prompt_length) >= self.max_tokens ):
      removed_chat_length = len(self.current_chat_history.pop(0).split(" "))
      self.current_chat_history_length -= removed_chat_length
    
    # print(json.dumps(self.current_chat_history, indent=2))
    prompt_messages = [{"role": "system", "content": self.initial_prompt}]
    prompt_messages.extend(self.current_chat_history)
    # print(prompt_messages)
    
    response = self.client.chat.completions.create(
      model=self.model,
      messages=prompt_messages,
      temperature=TEMPERATURE,
      top_p=TOP_P,
      # max_tokens=150,
    ).choices[0]
    
    # print(response)
    response_text = response.message.content
    print(response_text)
    response_text = self._filter_think(response_text)
    response_text = self._filter_emoji(response_text)
    
    response_length = len(response_text.split(" "))
    
    self.global_chat_history.append({
      "message": {"role": "user", "content": text},
      "length": text_length
    })
    self.global_chat_history.append({
      "message": {"role": "assistant", "content": response_text},
      "length": response_length
    })
    self.current_chat_history.append(
      {"role": "assistant", "content": response_text}
      )
    
    self._write_chat_history()
    
    logging.warning("🤖 Response returned")

    return response_text
    
    
    
    