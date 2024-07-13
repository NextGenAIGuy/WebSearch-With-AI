from langchain_community.utilities import SerpAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import json
import os
from dotenv import load_dotenv
load_dotenv()


serpapi_api_key = os.getenv("serpapi_api_key")
groq_api_key = os.getenv("groq_api_key")


search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")


prompt_template = """
    Answer the question based only on the context provided. 

    Question: {question}

    context: {context}
    """

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question","context"],
    )

chain = prompt | chat | StrOutputParser()

question = input("Please Enter your Query: ")

data = search.run(query=question)
result = chain.invoke({"question": question,"context": data})
print(result)