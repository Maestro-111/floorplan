import base64
import requests
import os


def encode_image(image_path):

  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')




def get_direction(image_scr, prompt, api_key):

    image_path1 = 'images/train/pdf_floor_plan_1_0.jpg.'
    image_path2 = 'images/train/pdf_floor_plan_1_1.jpg.'
    image_path3 = 'images/train/pdf_floor_plan_38_0.jpg.'
    image_path4 = 'images/train/pdf_floor_plan_38_1.jpg.'
    image_path5 = 'images/train/pdf_floor_plan_42_2.jpg.'
    image_path6 = 'images/train/pdf_floor_plan_19_3.jpg.'
    image_path7 = 'images/train/pdf_floor_plan_19_4.jpg.'
    image_path8 = 'images/train/pdf_floor_plan_42_0.jpg.'
    image_path9 = 'images/train/pdf_floor_plan_1_6.jpg.'
    image_path10 = 'images/train/pdf_floor_plan_1_10.jpg.'
    image_path11 = 'images/train/pdf_floor_plan_42_7.jpg.'
    image_path12 = 'images/train/pdf_floor_plan_42_8.jpg.'



    image_path13 = image_scr

    base64_image1 = encode_image(image_path1)
    base64_image2 = encode_image(image_path2)
    base64_image3 = encode_image(image_path3)
    base64_image4 = encode_image(image_path4)
    base64_image5 = encode_image(image_path5)
    base64_image6 = encode_image(image_path6)
    base64_image7 = encode_image(image_path7)
    base64_image8 = encode_image(image_path8)
    base64_image9 = encode_image(image_path9)
    base64_image10 = encode_image(image_path10)
    base64_image11 = encode_image(image_path11)
    base64_image12 = encode_image(image_path12)

    base64_image13 = encode_image(image_path13)


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {

        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image3}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image4}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image5}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image6}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image7}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image8}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image9}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image10}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image11}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image12}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image13}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 400,
        "temperature":0.1
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response



