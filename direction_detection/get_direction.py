from openai import OpenAI
import base64
import os
import requests
import shutil


def delete_files_in_directory(directory_path):
  files = os.listdir(directory_path)
  for file_name in files:
    file_path = os.path.join(directory_path, file_name)
    if os.path.isfile(file_path):
      os.remove(file_path)
      print(f"Deleted: {file_path}")
  print("All files deleted successfully.")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_GPT(img_path):



  base64_image = encode_image(img_path)


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
            "text": "there is an arrow besides the key plate object, pointing to the north."
                    "Given that, identify what is the direction of the unit window (north,east,west,south)?"
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

  lst = response.json()['choices'][0]['message']['content'].split(':')


  if len(lst) != 2:
    return 'No Direction Detected'

  return lst[1]



api_key=os.environ.get("OPENAI_API_KEY")


sample_path = 'C:/floorplan/sample'
test_path = 'test'

sample_images = os.listdir(sample_path)

for image in sample_images:
  shutil.copy(os.path.join(sample_path,image),os.path.join(test_path,image))



for image in os.listdir(test_path):
  print(f'Direction for {image} is' ,analyze_image_GPT(os.path.join(test_path, image)))

print()
delete_files_in_directory('test')

