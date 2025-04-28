import json
import sqlite3

from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate


def search_places_by_location(location, request_tags):
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM places WHERE location = ?", (location,))
    results = cursor.fetchall()
    scored_results = [(row, search_by_tags(row[3], request_tags)) for row in results]
    sorted_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
    accurate_results = [row for row, _ in sorted_results]
    print(accurate_results)
    return accurate_results


def search_by_tags(description_tags, request_tags):
    response_schemas = [
        {"name": "accuracy", "type": "int", "description": "How similar are the tags to each other? \
                                      Compare the tags from a place description and tags from request, match the \
                                      number  from 0.0 to 1.0 to the comparison, where 0.0 is 0 similar tags and 1.0 \
                                      are all tags from request are close to the description, \
                                      for each accuracy between given tags."}
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    review_template = """For the two following lists, compare the tags from a place description and tags from request: \
                accuracy: "How well are the tags overlap? \
                Compare the tags from a place description and tags from request, \
                match the number from 0.0 to 1.0 for each accuracy between given tags.\
                description_tags: {description_tags}, request_tags: {request_tags}, {format_instructions}"""

    prompt = ChatPromptTemplate.from_template(template=review_template)

    from langchain_community.chat_models import ChatOpenAI
    chat = ChatOpenAI(temperature=0.0, model="gpt-4")

    messages = prompt.format_messages(description_tags=description_tags, request_tags=request_tags,
                                      format_instructions=format_instructions)
    response = chat(messages)
    cleaned = response.content.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)
    accuracy = parsed["accuracy"]
    return accuracy
