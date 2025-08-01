from openai import OpenAI
import requests
import json

# client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

# response = client.completions.create(
#   model="orpheus-3b-0.1-ft",
#   prompt="<|Audio|>tara: Hello, are you having a great day?<|eot_id|>",
#   temperature=0.65,
#   top_p=0.9,
#   stream=True,
#   max_tokens=2048,
#   extra_body={ "repeat_penalty": 1.2 }
# )

# print(response)

# for chunk in response:
#   print(chunk.choices[0].text)



# Create the request payload for the LM Studio API
payload = {
    "model": "orpheus-3b-0.1-ft",  # Model name can be anything, LM Studio ignores it
    "prompt": "<|Audio|>tara: Hello, are you having a great day?<|eot_id|>",
    "max_tokens": 2048,
    "temperature": 0.65,
    "top_p": 0.9,
    "repeat_penalty": 1.2,
    "stream": True
}

# Make the API request with streaming
response = requests.post(
  url="http://127.0.0.1:1234/v1/completions",
  headers={"Content-Type": "application/json"},
  json=payload,
  stream=True)

if response.status_code != 200:
    print(f"Error: API request failed with status code {response.status_code}")
    print(f"Error details: {response.text}")
    quit()

# Process the streamed response
token_counter = 0
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data_str = line[6:]  # Remove the 'data: ' prefix
            if data_str.strip() == '[DONE]':
                break
                
            try:
                data = json.loads(data_str)
                if 'choices' in data and len(data['choices']) > 0:
                    token_text = data['choices'][0].get('text', '')
                    token_counter += 1
                    if token_text:
                        print(token_text)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue