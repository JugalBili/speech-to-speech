from openai import OpenAI
import logging
import json
import re
import os
from config import *

logger = logging.getLogger("speech_to_speech.llm_wrapper")

class LLMWrapper():
  def __init__(self):
    self.api = os.getenv("OPENAI_API")
    self.api_key = os.getenv("OPENAI_API_KEY")
    self.model = LLM_MODEL
    self.global_chat_history = []
    self.current_chat_history = []
    self.current_chat_history_length = 0
    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.chat_history_path = os.path.join(project_root_dir, "data", "chat_history.json")
    self.max_tokens = int(MAX_TOKENS * 0.75)
    
    self.initial_prompt = INITIAL_PROMPT
    self.initial_prompt += "Do not style your response using markdown formatting. Note that you responses must be in a conversation format, thus not text fomatting is allowed to make the output look nice after it has been rendered."
    if TTS_CHOICE == "orpheus":
      self.initial_prompt += " Also, add paralinguistic elements like <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp> or uhm for more human-like speech whenever it fits, but do not overdo it, please only add it when necessary and not often."
    if TTS_CHOICE == "kokoro":
      self.initial_prompt += "Here are some rules regarding how the output should be formatted such that it could work with text-to-speech. 1. To adjust intonation, try punctuation ;:,.!?—…\"()“” or stress ˈ and ˌ"
    self.initial_prompt = self.initial_prompt.replace("\n", "")
    self.initial_prompt_length = len(self.initial_prompt.split(" "))
    
    self.websearch_classifier_prompt = """
    You are a classifier that determines whether a user’s request requires an external web search, and if so, what the concise web search query should be.


    Rules for deciding:
    - Answer "yes" if the prompt is about facts, knowledge, history, current events, or time-sensitive information
    - Answer "no" if the request can be answered without external knowledge (general conversation, opinions, jokes, instructions, etc.).
    - If answering "yes", provide a corrected, concise search query containing only the essential topic and keywords limited to no more than 10 words.
    - If answering "no", the main topic should be "None".
    - Correct the spelling of words in the main topic if necessary as the input prompt it the output of speech-to-text, and the text will be slightly off.

    Output format (no extra words, no punctuation except as shown):  
    `<yes/no>+-+<main topic>`

    Examples:
    - "How tall is the Empire state?" → `yes+-+Empire State Height`
    - "What is the price of Bitcoin currently?" → `yes+-+Bitcoin Price`
    - "How are you doing today?" → `no+-+None`
    - "Explain what a black hole is." → `yes+-+What is Black Hole Explanation`
    - "Write me a poem about cats." → `no+-+None`
    - "Tell me the weather in Tokyo." → `yes+-+Tokyo Weather`
    """
    
    self.client = OpenAI(base_url=self.api, api_key=self.api_key)
    
    self._load_convo_history()
    
    
  def _load_convo_history(self):
    logger.debug("Loading conversation history")

    chat_history_file = open(self.chat_history_path, 'r', encoding="utf-8")
    self.global_chat_history = json.load(chat_history_file)["history"]
    chat_history_file.close()
    
    index = len(self.global_chat_history) - 1
    while index > -1 and ((self.current_chat_history_length + self.initial_prompt_length) < self.max_tokens ):
      cur_message = self.global_chat_history[index]
      self.current_chat_history.insert(0, cur_message["message"])
      self.current_chat_history_length += cur_message["length"]
      index -= 1
      
    # print(self.current_chat_history)


  def _write_chat_history(self):
    logger.debug(f"Saving conversation history")

    chat_history_file = open(self.chat_history_path, 'w', encoding="utf-8")
    json.dump({
        "history" : self.global_chat_history
      },
      chat_history_file
    )
    chat_history_file.close()


  def _filter_think(self, text):
    marker = "</think>"
    index = text.find(marker)
    
    # Think not found
    if index == -1:
      return text
    
    index = index + len(marker)
    filtered_text = text[index:].replace("\n", "")
    
    return filtered_text

  def _filter_markdown(self, text):
    # Remove code blocks
    filtered_text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code
    filtered_text = re.sub(r"`([^`]*)`", r"\1", filtered_text)
    # Remove bold/italic markers
    filtered_text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", filtered_text)
    filtered_text = re.sub(r"(\*|_)(.*?)\1", r"\2", filtered_text)
    # Remove headers
    filtered_text = re.sub(r"^#+\s*", "", filtered_text, flags=re.MULTILINE)
    # Remove blockquotes
    filtered_text = re.sub(r"^>\s*", "", filtered_text, flags=re.MULTILINE)
    # Remove horizontal rules
    filtered_text = re.sub(r"^-{3,}", "", filtered_text, flags=re.MULTILINE)

    return filtered_text.strip()
  
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

  def decide_websearch(self, text):
    prompt_messages = [
      {"role": "system", "content": self.websearch_classifier_prompt},
      {"role": "user", "content": text + "/no_think"}
    ]
    
    response = self.client.chat.completions.create(
      model=self.model,
      messages=prompt_messages,
      temperature=TEMPERATURE,
      top_p=TOP_P,
      # max_tokens=150,
    ).choices[0]
    
    response_text = response.message.content
    response_text = self._filter_think(response_text)
    response_text = self._filter_emoji(response_text)
    response_text = self._filter_markdown(response_text)
    
    require_search, topic = response_text.split("+-+")

    return require_search.lower(), topic.lower()


  def send_to_llm(self, text, context = ""):
    if not ENABLE_THINK and "instruct" not in LLM_MODEL:
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
    
    # if context exists, add to user prompt
    if context != "":
      prompt_messages[-1]["content"] =  "<context>"+ context.replace("\n", "") + "</context>\n\n" + prompt_messages[-1]["content"]
    
    # print(prompt_messages)
    
    response = self.client.chat.completions.create(
      model=self.model,
      messages=prompt_messages,
      temperature=TEMPERATURE,
      top_p=TOP_P,
      # max_tokens=150,
    ).choices[0]
    
    response_text = response.message.content
    response_text = self._filter_think(response_text)
    response_text = self._filter_emoji(response_text)
    response_text = self._filter_markdown(response_text)
    
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
    
    logger.debug("Response returned")

    return response_text
    
    
    
    