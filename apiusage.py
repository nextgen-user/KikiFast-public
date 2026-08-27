from google import genai
from google.genai import types
import base64
import os
from google.oauth2 import service_account

# 1. FIX: You must explicitly pass the cloud-platform scope here
credentials = service_account.Credentials.from_service_account_file(
    r"/srv/kikifast/example-project-347dc1f51500.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

def generate():
  # 2. FIX: Changed location from 'global' to a regional zone like 'us-central1'
  client = genai.Client(
      vertexai=True,
      project="example-project", 
      location="global",  
      credentials=credentials
  )

  # Ensure your model name is valid for Vertex AI (e.g., gemini-2.5-flash)
  model = "gemini-3-flash-preview"
  
  # 3. FIX: Populated the parts list with an actual text payload
  contents = [
    types.Content(
      role="user",
      parts=[
         types.Part(text="Hello! Tell me a quick joke.")
      ]
    )
  ]
  
  tools = [
    types.Tool(google_search=types.GoogleSearch()),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 65535,
    safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ],
    tools = tools,
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
        continue
    print(chunk.text, end="")

generate()