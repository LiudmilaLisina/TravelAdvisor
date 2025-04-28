import json

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from chain import overall_chain, format_instructions
from find_place_by_location_and_tags import search_places_by_location

import warnings

warnings.filterwarnings("ignore")

from load_bd import load_bd

load_bd()

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")


@tool
def extract_trip_destination(review: str) -> str:
    """Search and recommend **specific places** (hotels, cities, regions) based on a travel request. Use this tool
    if the user mentions anything about:
    - where to go
    - cities
    - hotels
    - activities
    - budget
    - companions

    This tool performs a structured search in a real database and MUST be used whenever a destination or place is implied.
    """
    formated_request = overall_chain.invoke({
        "request": review,
        "format_instructions": format_instructions
    })
    cleaned = ''.join(formated_request['details_review'].replace("```json", "").replace("```", "").split(" "))
    parsed = json.loads(cleaned)
    location = parsed["location"]
    tags = parsed["tags"]
    found_places = search_places_by_location(location, tags)
    if len(found_places) == 0:
        return "Unfortunately, nothing was found for your search. Try to change the criteria."
    response = "Based on your preferences, here are the matching destinations:\n\n"
    for _, name, location, tags, _ in found_places:
        tag_summary = tags
        response += f"• **{name}** ({location}) — {tag_summary}\n"

    response += "\nThese places were selected strictly from our database using your preferences. No other data was added."
    return "Describe why are these places worth visit" + response


@tool
def general_qa() -> str:
    """Answer the general travel questions. Use this for any question that does NOT require choosing or
    analyzing a travel destination."""
    return "Here's a quick answer to your travel question! Let me know when you're ready to search for a destination."


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you're a helpful travel assistant",),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

from langchain_core.tools import Tool

tools = [
    Tool.from_function(
        func=extract_trip_destination,
        name="extract_trip_destination",
        description="Use this tool if the user asks about places to go, cities, hotels, regions, or if the request "
                    "involves location, budget, companions, or amenities.",
        return_direct=False
    ),
    Tool.from_function(
        func=general_qa,
        name="general_qa",
        description="Answer travel-related questions only if they do NOT ask for places, hotels, destinations, "
                    "or anything location-based.",
        return_direct=False
    )
]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# request = ("I'm planning a budget trip to Hamburg with my fiancé. I care about hotels with fridges and private "
#            "bathrooms. What do you recommend?")
#
# request = ("I want to go to Hamburg")
# result = agent_executor.invoke({
#     "input": request
# })
#
# print(result["output"])
