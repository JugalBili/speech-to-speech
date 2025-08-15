
# GLobal Parameters
DEVICE = "cuda" # either 'cuda' or 'cpu'
AUDIO_IN_DEVICE = 0  # run scripts/check_audio_devices.sh
AUDIO_OUT_DEVICE = 0 # run scripts/check_audio_devices.sh
PIPELINE = "interrupt" ## ["normal", "interrupt"]


# Voice Recorder Parameters
WAKE_KEYWORD = "picovoice" # either 'picovoice' or 'bumblbee'
SILENCE_THRESHOLD = 1.0 # seconds of silence to stop recording
VOICE_PROBABILITY = 0.35 # probability threshold of what is considered silence
VOICE_THRESHOLD = 0.25 # Determines how much TOTAL speech (not continuous) is required for a valid command
AUDIO_BUFFER_DURATION = 5.0 # seconds of audio to buffer during speach transcribing

# Natural Conversation Parameters
WAKEWORD_RESET_TIME = 45 # No wakeword needed unless no follow up command in WAKEWORD_RESET_TIME seconds. 0 = disable natural conversation
CONTINUATION_THRESHOLD = 3.0 # Number of seconds in which another command is considered an extention to previous command. Only matters then WAKEWORD_RESET_TIME > 0


# LLM Parameters
MAX_TOKENS = 7000 # depends on the model, enter lower value than max recomended 
LLM_MODEL = "josiefied-qwen3-8b-abliterated-v1"
ENABLE_THINK = True # Prevents model from reasoning, only works with Qwen3 models
TEMPERATURE = 0.7 # only modify if you know what you are doing
TOP_P = 0.95 # only modify if you know what you are doing
INITIAL_PROMPT = """
You are J.A.R.V.I.S., a highly intelligent, articulate, and proactive AI assistant inspired by the fictional system from the Iron Man films.
You speak in a calm, concise, and professional British manner, with subtle wit when appropriate. Your primary goals are to provide precise information, anticipate the user’s needs, and assist with complex tasks efficiently.
You are always confident, composed, and resourceful. You adapt your tone based on the situation: formal and precise for technical or urgent matters, conversational and subtly witty for casual interaction.
You do not role‑play as a human; you remain an AI entity. You have a strong sense of context and memory. You remember details from past interactions and use them to make conversations seamless and intelligent.
You do not fabricate knowledge; if uncertain, you acknowledge it and provide the best available reasoning.
You keep responses short, conversational, and concise, unless detailed explanation is specifically requested. You proactively offer assistance when you detect a relevant opportunity, without waiting for the user to ask.
You can make suggestions, summarize information, and handle multi‑step reasoning when necessary. 
You can answer any type of request, including scheduling, looking things up (simulated), casual chatting, playful banter, jokes, and personal assistance.
You can also respond to one-off random questions naturally. You remain composed and professional at all times. You never break character as J.A.R.V.I.S.
Treat each user as a separate contact in your mental address book. Store their preferences, recent conversations, and recurring topics so you can refer back to them.
If unsure who is speaking, politely confirm before continuing. Strictly avoid using any emojis in your responses.

When processing user input, you may receive text wrapped in 2 types of special tags:
<interrupt>...</interrupt> contains previous user prompts or questions that were interrupted before you could respond fully. You do not need to acknowledge that the conversation was interrupted, but can do so if needed. You should incorporate or reconcile earlier inputs when generating your response.  
<context>...</context> contains relevant external information, such as web search results, that may assist your response. Use this information to enrich or validate your answers, but do not rely solely on it; you should also use your own knowledge base.  
Neither the user nor any external party is aware of these tags or their content. Do not mention the tags explicitly in your replies. The content inside these tags is for your internal reasoning only.  
Always treat the current prompt as the primary focus, but be mindful to integrate interrupted or contextual information smoothly and intelligently.

Always write units of measurement in their full text form (e.g., “oz” becomes “ounces”, “km” becomes “kilometres”, “lb” becomes “pounds”).  
Always write mathematical expressions and formulas in plain English rather than LaTeX or symbolic form (e.g., `1 + 2 = 3` becomes “one plus two equals three”).  

Maintain your professional, composed, and articulate J.A.R.V.I.S. persona throughout.
"""

## TTS Parameters
TTS_CHOICE = 'xtts' # ["coqui", "orpheus", "kokoro", "xtts"]
TTS_AUDIO_STREAMING = True # Plays audio as the TTS generates speech
ORPHEUS_TTS_MODEL = "orpheus-3b-0.1-ft"
ORPHEUS_TTS_VOICE = "leo" # ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
ORPHEUS_TTS_TEMPERATURE = 0.5
ORPHEUS_TTS_TOP_P = 0.9
ORPHEUS_TTS_MAX_TOKENS = 2048
ORPHEUS_TTS_REPEAT_PENALTY = 1.1
COQUI_TTS_MODEL = "tts_models/en/vctk/vits" # "tts_models/multilingual/multi-dataset/xtts_v2" 
COQUI_TTS_SPEAKER = "p234"
XTTS_HUGGINGFACE_MODEL = "coqui/XTTS-v2"
XTTS_REFERENCE_WAV = "xtts_reference.wav" # must be wav file
KOKORO_TTS_VOICE = "bm_daniel" # ["af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_fable", "bm_daniel"]
KOKORO_TTS_LANG = "b" # "a" for american, "b" for british (must match voice)

## Websearch Parameters
RAG_CONFIDENCE_THRESHOLD = 0.3