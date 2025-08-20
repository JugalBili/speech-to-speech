from openai import OpenAI
import logging
import json
import re
import os
import time
from datetime import datetime
from config import *
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass


class LLMWrapper():
  def __init__(self, interrupt_count:SynchronizedClass, memories = []):
    self.logger = logging.getLogger("speech_to_speech.llm_wrapper")
    self.interrupt_count = interrupt_count
    self.interrupt_context = []

    self.api = os.getenv("OPENAI_API")
    self.api_key = os.getenv("OPENAI_API_KEY")
    self.model = LLM_MODEL
    self.global_chat_history = []
    self.current_chat_history = []
    self.current_chat_history_length = 0
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    self.chat_history_path = os.path.join(project_root_dir, "data", "chat_history.json")
    self.max_tokens = int(MAX_TOKENS * 0.75)
    
    self.memories = memories
    self.initial_prompt = INITIAL_PROMPT
    self.initial_prompt += "Do not style your response using markdown formatting. Note that you responses must be in a conversation format. Thus no text fomatting is allowed to make the output look nice after it has been rendered."
    if TTS_CHOICE == "orpheus":
      self.initial_prompt += " Also, add paralinguistic elements like <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp> or uhm for more human-like speech whenever it fits, but do not overdo it, please only add it when necessary and not often."
    if TTS_CHOICE == "kokoro":
      self.initial_prompt += "Here are some rules regarding how the output should be formatted such that it could work with text-to-speech. 1. To adjust intonation, try punctuation ;:,.!?—…\"()“” or stress ˈ and ˌ"
    self.initial_prompt = self.initial_prompt.replace("\n", "")
    self.initial_prompt_length = len(self.initial_prompt.split(" "))
    
    self.websearch_memory_classifier_prompt = """
    You are a classifier that performs TWO tasks for each user prompt.    

    1. Web Search Classification: determine whether a user’s request requires an external web search, and if so, what the concise web search query should be.

    Web Search Classification Rules:
    - Answer "yes" if the prompt explicitly asks about or references uncommon facts, knowledge, history, current events, or time-sensitive information
    - Answer "no" if the request can be answered without external knowledge (general conversation, opinions, jokes, instructions, etc.).
    - If answering "yes", provide a corrected, concise search query containing only the essential topic and keywords limited to no more than 10 words.
    - If answering "no", the main topic should be "None".
    - Correct the spelling of words in the main topic if necessary as the input prompt it the output of speech-to-text, and the text will be slightly off.

    Web Search Classification Output format (no extra words, no punctuation except as shown):  
    `<yes/no>+-+<main topic>`

    Web Search Classification Examples:
    - "How tall is the Empire state?" → `yes+-+Empire State Height`
    - "What is the price of Bitcoin currently?" → `yes+-+Bitcoin Price`
    - "How are you doing today?" → `no+-+None`
    - "Explain what a black hole is." → `yes+-+What is Black Hole Explanation`
    - "Write me a poem about cats." → `no+-+None`
    - "Tell me the weather in Tokyo." → `yes+-+Tokyo Weather`

    2. Deep Memory Extraction: determines whether a user’s prompt is contains information worth remembering in deep memory.

    Deep Memory Extraction Rules:
    - Extract directly, or indirectly information about the user or what the user explicitly asks the system to remember.
    - This includes Names, relationships, preferences, goals, identity info.
    - Ignore casual conversation, temporary context, or general questions.
    - If nothing important should be remembered, return "None".

    Deep Memory Extraction Output format (no extra words, no punctuation except as shown):
    - <deep_memory:...> where "..." is replaced by the information about the user to remember or "None"

    Deep Memory Extraction Examples:
    - "How tall is the Empire state?" → <deep_memory:None>
    - "My investments are not doing well right now" → <deep_memory:None>
    - "What is the price of Bitcoin currently?" → <deep_memory:None>
    - "My friend Bob sold me that I am short, is he right?" → <deep_memory:Users best friend is Bob>
    - "Bob is wearing a red jacket and has black hair" → <deep_memory:Bob wears a red jacket and has black hair>
    - "Call me Jeff" → <deep_memory:User wants to be called Jeff>
    - "Tell me the weather in Tokyo." → <deep_memory:None>
    - "I like cheese" → <deep_memory:User likes cheese>
    - "I am currently streaming on twitch right now" - <deep_memory:User streams on twitch>


    Overall output structure: Should only consists of 2 lines
    - First line = websearch classifier result
    - Second line = memory extraction result
    """
    
    self.client = OpenAI(base_url=self.api, api_key=self.api_key)
    
    self._load_convo_history()
    
    
  def _load_convo_history(self):
    self.logger.debug("Loading conversation history")

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
    self.logger.debug(f"Saving conversation history")

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

  def decide_websearch_memory(self, text):
    prompt_messages = [
      {"role": "system", "content": self.websearch_memory_classifier_prompt},
      {"role": "user", "content": text + "" if "instruct" in LLM_MODEL else "/no_think"}
    ]

    response_text = ""
    
    stream = self.client.chat.completions.create(
      model=self.model,
      messages=prompt_messages,
      temperature=TEMPERATURE,
      top_p=TOP_P,
      stream=True
      # max_tokens=150,
    )

    for chunk in stream:
      if self.interrupt_count.value > 0:
        # On interrupt
        return None, None
      
      data = chunk.choices[0].delta.content
      if data is not None:
        response_text += data

    response_text = self._filter_think(response_text)
    response_text = self._filter_emoji(response_text)
    response_text = self._filter_markdown(response_text)
    
    websearch_line = response_text.split("\n")[0]
    memory_line = response_text.split("\n")[-1]
    
    require_search, topic = websearch_line.split("+-+")
    memory = None
    
    memory_match = re.search(r"<deep_memory:(.*?)>", memory_line)
    if memory_match:
      memory = memory_match.group(1)

    return (require_search.lower(), topic.lower()), memory


  def send_to_llm(self, text, timestamp, memory = "None", context = ""):
    if not ENABLE_THINK:
      text = text + " /no_think" # disable reasoning

    text_length = len(text.split(" "))
    
    if memory != "None":
      self.memories.append(f"{datetime.fromtimestamp(timestamp).strftime("%m-%d-%y %H:%M:%S")} - {memory}")
    
    memories_text = ""
    for memory in self.memories:
      memories_text += f"<memory>{memory}</memory>\n"

    init_prompt_mem = self.initial_prompt + "\n\n" + memories_text
    init_prompt_mem_length = len(init_prompt_mem.split(" "))


    prompt_modification = ""
    # if interrupted, let LLM know
    if len(self.interrupt_context) > 0:
      for entry in self.interrupt_context:
        prompt_modification += "<interrupt>"+ entry + "</interrupt>\n"
        
    interrupt_text = prompt_modification + text
    interrupt_text_length = len(interrupt_text.replace("\n", " ").split(" "))
    
    self.current_chat_history.append(
      {"role": "user", "content": interrupt_text}
    )
    
    self.current_chat_history_length += interrupt_text_length
    
    while self.current_chat_history and ((self.current_chat_history_length + init_prompt_mem_length) >= self.max_tokens ):
      removed_chat_length = len(self.current_chat_history.pop(0).split(" "))
      self.current_chat_history_length -= removed_chat_length
    
    # print(json.dumps(self.current_chat_history, indent=2))
    prompt_messages = [{"role": "system", "content": init_prompt_mem}]
    prompt_messages.extend(self.current_chat_history)
    
    # if context exists, add to user prompt
    if context != "":
      prompt_modification = "<context>"+ context.replace("\n", "") + "</context>\n\n"
    
    prompt_messages[-1]["content"] =  prompt_modification + prompt_messages[-1]["content"]
    
    # print(prompt_messages)
    response_text = ""
    
    stream = self.client.chat.completions.create(
      model=self.model,
      messages=prompt_messages,
      temperature=TEMPERATURE,
      top_p=TOP_P,
      stream=True
      # max_tokens=150,
    )
    for chunk in stream:
      if self.interrupt_count.value > 0:
        # On interrupt
        return None

      data = chunk.choices[0].delta.content
      if data is not None:
        response_text += data
    response_timestamp = time.time()
    response_text = self._filter_think(response_text)
    response_text = self._filter_emoji(response_text)
    response_text = self._filter_markdown(response_text)
    
    response_length = len(response_text.split(" "))
    
    self.global_chat_history.append({
      "message": {"role": "user", "content": interrupt_text},
      "length": interrupt_text_length,
      "timestamp": timestamp
    })
    self.global_chat_history.append({
      "message": {"role": "assistant", "content": response_text},
      "length": response_length,
      "timestamp": response_timestamp
    })
    self.current_chat_history.append(
      {"role": "assistant", "content": response_text}
    )
    
    self._write_chat_history()
    
    self.logger.debug("Response returned")

    return response_text
    
    
    
    