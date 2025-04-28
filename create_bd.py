import json
import sqlite3

import pandas as pd
from langchain_community.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.0, model="gpt-4")

from langchain.output_parsers import StructuredOutputParser

from langchain.prompts import ChatPromptTemplate


def create_bd(data_base):
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    df = pd.read_csv('reviews.csv')
    names = df['Place'].tolist()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS places (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        location TEXT,
        tags TEXT,
        review TEXT
    )
    ''')
    for i in range(len(names)):
        cursor.execute('''
        INSERT INTO places (name, location, tags, review)
        VALUES (?, ?, ?, ?)
        ''', (
            df.at[i, 'Place'],
            df.at[i, 'Location'],
            json.dumps(get_csv(df.at[i, 'Reviews'])["tags"]),
            df.at[i, 'Reviews']
        ))
    conn.commit()


def get_csv(review):
    response_schemas = [
        {"name": "tags", "type": "list", "description": "Tags for a place search in a data base. \
                                  Extract any sentences about the atmosphere, special services required, \
                                  kids or pets mentioned and other important details. \
                                  Give them as a comma separated Python list, shorten the detains\
                                  if this information is not found, output []."}
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    review_template = """For the following text, extract the following information: \
            tags: Tags for a place search in a data base. \
            Make them as useful for search as possible. \
            If this information is not found, output []. \
            text: {review}, {format_instructions}"""

    prompt = ChatPromptTemplate.from_template(template=review_template)

    messages = prompt.format_messages(review=review, format_instructions=format_instructions)
    response = chat(messages)
    return output_parser.parse(response.content)
