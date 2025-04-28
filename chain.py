from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())

import warnings

warnings.filterwarnings("ignore")

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

place_sсhema = ResponseSchema(name="place",
                              description="The type of the place to go on a trip .\
                              about the type of the place,\
                             if this information is not found, output\
                             Unknown.")
location_schema = ResponseSchema(name="location",
                                 description="Location of the travel destination.\
                             Extract any sentences about the city, country, where the user wants to go,\
                             if this information is not found, output\
                             Unknown.")
tags_schema = ResponseSchema(name="tags",
                             description="Tags for a place search in a data base. \
                              Extract any sentences about the atmosphere, special services required, \
                              kids or pets mentioned and other important details. \
                              Give them as a comma separated Python list, shorten the detains\
                              if this information is not found, output [].")

response_schemas = [place_sсhema,
                    location_schema,
                    tags_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

from langchain.chains import SequentialChain
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    """For the following text, extract the following information: \
    companions: Who also was on the vacation? \
    Information about the people, who were on the trip \
    and output them as a comma separated string. \
    type of the place: Why kind of place is user looking for? \
    Information about the type of the place (hotel, cafe, museum and etc.), \
    if this information is not found, output Unknown. \
    location: Where the user wants to go? \
    If this information is not found, output Unknown.\
    tags: Tags for a place search in a data base. \
    Make them as useful for search as possible. \
    If this information is not found, output []. \
    text: {request}, {format_instructions}"""
)

chain_one = LLMChain(llm=llm, prompt=first_prompt,
                     output_key="details_review",
                     )

# second_prompt = ChatPromptTemplate.from_template(
#     "Can you summarize the following review in 1 sentence using the details:"
#     "\n\n{Details_Review}"
# )

# chain_two = LLMChain(llm=llm, prompt=second_prompt,
#                      output_key="summary"
#                     )
#
# third_prompt = ChatPromptTemplate.from_template(
#     "What is the most relevant detail from this review:\n\n{Review}"
# )
#
# chain_three = LLMChain(llm=llm, prompt=third_prompt,
#                        output_key="detail"
#                       )
#
# fourth_prompt = ChatPromptTemplate.from_template(
#     "Write a follow up response to the following "
#     "summary paying attention on the key detail:"
#     "\n\nSummary: {summary}\n\nDetail: {detail}"
# )
# # chain 4: input= summary, language and output= followup_message
# chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
#                       output_key="followup_message"
#                      )

overall_chain = SequentialChain(
    chains=[chain_one],
    # chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["request", "format_instructions"],
    # output_variables=["details_review", "summary","followup_message"],
    output_variables=["details_review"],
    verbose=True
)
