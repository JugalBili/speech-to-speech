from openai import OpenAI
import time

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

prompt = """
test

/no_think
"""

prompt_messages = [
  {
    "role": "user",
    "content": prompt}
]

start_time_perf = time.perf_counter()
response = client.chat.completions.create(
  model="josiefied-qwen3-0.6b-abliterated-v1",
  messages=prompt_messages,
  temperature=0.7,
  top_p=0.95,
  # max_tokens=150,
).choices[0]
end_time_perf = time.perf_counter()


print(response.message.content)
print(end_time_perf - start_time_perf)