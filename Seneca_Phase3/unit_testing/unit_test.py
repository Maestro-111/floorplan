import pandas as pd
import numpy as np

import openai
from openai import OpenAI

import os

import re

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)



database = pd.read_excel("database.xlsx")
craft_ocr_output = pd.read_excel("craft_ocr_output.xlsx")
craft_ocr_output.drop(columns=['Unnamed: 0'], inplace=True)


joined_dfs = database.merge(craft_ocr_output, on='Name', how='inner')

with open('prompt_area.txt', 'r') as file:
    message_area_template = file.read()


with open('prompt_rooms.txt', 'r') as file:
    message_rooms_template = file.read()

with open('prompt_floors.txt', 'r') as file:
    message_floors_template = file.read()

result_df = joined_dfs[['Name','Inner','Outer','Dens','Bedrooms','Bathrooms','Floors']]


data_inners = []
data_outers = []
data_area_prompts = []
data_room_prompts = []
data_floor_promps = []
data_dens = []
data_bathrooms = []
data_bedrooms = []
data_floors = []

for row in joined_dfs.iterrows():

    ocr_out = row[1]['Response']
    file_name = row[1]['Name']

    ############# AREA

    res = message_area_template.format(len(ocr_out), ocr_out)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": res,
            }
        ],
        model="gpt-4",
        temperature=1,
    )

    resp = chat_completion.choices[0].message.content
    resp = resp.split('\n')
    resp = [i for i in resp if i]

    iner_gpt_output = 0
    outer_gpt_output = 0

    if len(resp) != 2:
        print("Could not extract area info\n")

        data_inners.append(iner_gpt_output)
        data_outers.append(outer_gpt_output)

    else:
        inner, total = resp

        inner = inner.split(":")
        total = total.split(":")

        inner = inner[1]
        total = total[1]

        total = re.findall(r'(?:\d+[,.]?)+', total)
        inner = re.findall(r'(?:\d+[,.]?)+', inner)

        if len(inner) == 0:
            pass

            if len(total) == 0:
                pass
            else:
                total = total[0]
                total = int(''.join([ch for ch in total if ch.isdigit()]))

        else:
            inner = inner[0]
            inner = int(''.join([ch for ch in inner if ch.isdigit()]))

            iner_gpt_output = inner


            if len(total) == 0:
                pass
            else:
                total = total[0]
                total = int(''.join([ch for ch in total if ch.isdigit()]))
                outer_gpt_output = total - inner

        data_inners.append(iner_gpt_output)
        data_outers.append(outer_gpt_output)

    data_area_prompts.append(res)

    ########## ROOM

    res = message_rooms_template.format(ocr_out)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": res,
            }
        ],
        model="gpt-4",
        temperature=1,
    )

    resp = chat_completion.choices[0].message.content
    resp = resp.split('\n')
    resp = [i for i in resp if i]

    dens_gpt_output = 0
    bathrooms_gpt_output = 0
    bedrooms_gpt_output = 0


    if len(resp) != 3:
        print("Could not extract area info\n")
        data_dens.append(dens_gpt_output)
        data_bathrooms.append(bathrooms_gpt_output)
        data_bedrooms.append(bedrooms_gpt_output)
    else:
        bedrooms,dens,bathrooms=resp

        bedrooms = int(bedrooms.split(":")[1])
        dens = int(dens.split(":")[1])
        bathrooms = int(bathrooms.split(":")[1])

        dens_gpt_output = dens
        bathrooms_gpt_output = bathrooms
        bedrooms_gpt_output = bedrooms

        data_dens.append(dens_gpt_output)
        data_bathrooms.append(bathrooms_gpt_output)
        data_bedrooms.append(bedrooms_gpt_output)

    data_room_prompts.append(res)

    ####### FLOORS

    res = message_floors_template.format(ocr_out)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": res,
            }
        ],
        model="gpt-4",
        temperature=1,
    )


    resp = chat_completion.choices[0].message.content
    resp = resp.split('\n')
    resp = [i for i in resp if i]
    floors_gpt_output = '0-0'

    if len(resp) != 1:
        data_floors.append(floors_gpt_output)
    else:
        floors = resp[0]
        name,value = floors.split(":")
        floors_gpt_output = value
        data_floors.append(floors_gpt_output)

    data_floor_promps.append(res)




result_df['Inner_GPT'] = data_inners
result_df['Outer_GPT'] = data_outers

result_df['Dens_GPT'] = data_dens
result_df['Bathrooms_GPT'] = data_bathrooms
result_df['Bedrooms_GPT'] = data_bedrooms

result_df['Floors_GPT'] = data_floors

result_df['Prompt Area Message'] = data_area_prompts
result_df['Prompt Rooms Message'] = data_room_prompts
result_df['Prompt Floor Message'] = data_floor_promps

result_df['Floors'] = result_df['Floors'].astype(str)
result_df['Floors_GPT'] = result_df['Floors_GPT'].astype(str)

result_df['Floors'] = result_df['Floors'].apply(lambda x: x.strip().lower())
result_df['Floors_GPT'] = result_df['Floors_GPT'].apply(lambda x: x.strip().lower())



result_df['Area Error'] = np.where((result_df['Inner'] == result_df['Inner_GPT']) & (result_df['Outer'] == result_df['Outer_GPT']), 0, 1)
print(f"Propotion of area mistakes: {round(sum(result_df['Area Error'])/len(result_df['Area Error']),2)*100}%")
result_df_filtered_errors = result_df[result_df['Area Error'] == 1]
result_df_filtered_errors[['Name','Inner','Outer','Inner_GPT','Outer_GPT','Prompt Area Message']].to_excel("area_errors.xlsx")

result_df['Room Error'] = np.where((result_df['Dens'] == result_df['Dens_GPT']) & (result_df['Bathrooms'] == result_df['Bathrooms_GPT'] &
                                                                                   (result_df['Bedrooms'] == result_df['Bedrooms_GPT'])), 0, 1)
print(f"Propotion of rooms mistakes: {round(sum(result_df['Room Error'])/len(result_df['Room Error']),2)*100}%")
result_df_filtered_errors = result_df[result_df['Room Error'] == 1]
result_df_filtered_errors[['Name','Dens','Bedrooms','Bathrooms','Dens_GPT','Bedrooms_GPT','Bathrooms_GPT','Prompt Rooms Message']].to_excel("room_errors.xlsx")


result_df['Floor Error'] = np.where(result_df['Floors'] == result_df['Floors_GPT'], 0, 1)
print(f"Propotion of floor mistakes: {round(sum(result_df['Floor Error'])/len(result_df['Floor Error']),2)*100}%")
result_df_filtered_errors = result_df[result_df['Floor Error'] == 1]
result_df_filtered_errors[['Name','Floors','Floors_GPT','Prompt Floor Message']].to_excel("floor_errors.xlsx")