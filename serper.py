from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import json
from dotenv import load_dotenv
import os


load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
groq_api_key = os.getenv("groq_api_key")
    

llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")



prompt_template = """
    Answer the question based only on the context provided. 

    Context: {context}

    Question: {question}
    """

# Initialize LangChain components
search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY, k=3)

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question","context"],
    )


chain_version = prompt | llm | StrOutputParser()

question = input("Please Enter your Query: ")

search_output = search.run(query=question)
print(f"--------{search_output}----------------")

result = chain_version.invoke({"question": question,"context": search_output})
print(result)
                
#img_data = search.results(query=technology_name, type='images')
#print(img_data)

            
