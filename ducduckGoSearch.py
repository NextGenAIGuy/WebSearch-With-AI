from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("groq_api_key")
search = DuckDuckGoSearchResults()

Question = input("Enter your Query: ")
search_result = search.run(Question)
print(f"search result ----------->  {search_result} ---------------")


chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt_template = """
    Answer the question based only on the context provided.

    Context: {Context}

    Question: {Question}
    """

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["Context","Question"],
    )

chain = prompt | chat | StrOutputParser()


result = chain.invoke({"Context": search_result,"Question": Question})

print(result)