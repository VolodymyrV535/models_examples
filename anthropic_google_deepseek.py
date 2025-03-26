# imports
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai


# Print the key prefixes to help with any debugging

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

if deepseek_api_key:
    print(f"Deepseek API Key exists and begins {deepseek_api_key[:8]}")
else:
    print("Deepseek API Key not set")
    
    
# Connect to OpenAI, Anthropic
openai = OpenAI()
claude = anthropic.Anthropic()

# This is the set up code for Gemini
google.generativeai.configure()
deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")


## Asking LLMs to tell a joke
system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]


# GPT-3.5-Turbo
print("GPT-3.5-Turbo:")

completion = openai.chat.completions.create(model='gpt-3.5-turbo', messages=prompts)
print(completion.choices[0].message.content)

# GPT-4o-mini
# Temperature setting controls creativity
print("GPT-4o-mini")

completion = openai.chat.completions.create(
    model='gpt-4o-mini',
    messages=prompts,
    temperature=0.7
)
print(completion.choices[0].message.content)


# Claude 3.5 Sonnet
# API needs system message provided separately from user prompt
# Also adding max_tokens
print("Claude 3.5 Sonnet:")

message = claude.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

print(message.content[0].text)


# Claude 3.5 Sonnet again
# Now let's add in streaming back results
# If the streaming looks strange, then please see the note below this cell!
print("with streaming...")

result = claude.messages.stream(
    model="claude-3-5-sonnet-latest",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

with result as stream:
    for text in stream.text_stream:
            print(text, end="", flush=True)


# The API for Gemini has a slightly different structure.
print("Gemini:")

gemini = google.generativeai.GenerativeModel(
    model_name='gemini-2.0-flash-exp',
    system_instruction=system_message
)
response = gemini.generate_content(user_prompt)
print(response.text)


# As an alternative way to use Gemini that bypasses Google's python API library,
# Google has recently released new endpoints that means you can use Gemini via the client libraries for OpenAI!
print("Gemini via openai client:")

gemini_via_openai_client = OpenAI(
    api_key=google_api_key, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = gemini_via_openai_client.chat.completions.create(
    model="gemini-2.0-flash-exp",
    messages=prompts
)
print(response.choices[0].message.content)


# Using DeepSeek Chat
print("Deepseek:")
deepseek_via_openai_client = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
)

response = deepseek_via_openai_client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)


# Using DeepSeek Chat with a harder question! And streaming results
print("Deepseek with streaming...")

challenge = [{"role": "system", "content": "You are a helpful assistant"},
             {"role": "user", "content": "How many words are there in your answer to this prompt"}]

stream = deepseek_via_openai_client.chat.completions.create(
    model="deepseek-chat",
    messages=challenge,
    stream=True
)

reply = ""
for chunk in stream:
    reply = chunk.choices[0].delta.content or ''
    reply = reply.replace("```","").replace("markdown","")
    print(reply, end="")

print("Number of words:", len(reply.split(" ")))


# Using DeepSeek Reasoner - this may hit an error if DeepSeek is busy
print("Deepseek reasoner:")

response = deepseek_via_openai_client.chat.completions.create(
    model="deepseek-reasoner",
    messages=challenge
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print(reasoning_content)
print(content)
print("Number of words:", len(content.split(" ")))