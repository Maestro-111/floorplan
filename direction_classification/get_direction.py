from openai import OpenAI
import base64
import os
import requests

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


api_key=os.environ.get("OPENAI_API_KEY")


image_path = "pdf_floor_plan_1_3.jpg"
base64_image = encode_image(image_path)


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}


payload = {
  "model": "gpt-4-turbo",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Given this floor plan image, what is the specified direction (north,east,west or south)?"
                  "Give your response STRICTLY in this format: direction:your answer"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}


response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
print(response.json())


