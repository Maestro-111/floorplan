'''
import spacy
from spacy.lang.en.examples import sentences

nlp1 = spacy.load("en_core_web_sm")
nlp2 = spacy.load("en_core_web_trf")

text = "COCOA BEACH aeo sa.rt. STUDIO + 145 SQ.FT. TERRACE = 605 SQ.FT. TOTAL LIVING SPACE".lower()

# Process the text with spaCy
doc = nlp2(text)

# Iterate over tokens and print POS tags
for token in doc:
    print(token.text, token.pos_)


text = "BALCONY 30 SQ.FT.".lower()

print("!")

# Process the text with spaCy
doc = nlp2(text)

# Iterate over tokens and print POS tags
for token in doc:
    print(token.text, token.pos_)
'''

import os
import openai
from openai import OpenAI

a = f"Given these 2 strings derived from the floor plan: "
b = f"two bedroom 805 sq.ft."
c = f"; balcony 56 sq.ft. "
d = f"determine which string describes the total area (e.g which string contains total area) and in that string get that total area."
e = f"and then give your response in this format: total area : found area. Remember we are not looking for the total area" \
    f" - we are looking for the string that contains the total area number and we want to derive that number"

res = a+b+c+d+e
print(res)

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": res,
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message.content)