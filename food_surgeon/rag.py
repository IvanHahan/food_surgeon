
from dotenv import load_dotenv
from langchain import hub
from langchain.output_parsers import PydanticToolsParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from food_surgeon.db import get_vector_db

load_dotenv(".env")


class Dish(BaseModel):
    """Always use this tool to structure your response to the user."""
    id: str = Field(description="Identifier of the dish from the retrieved data.")
    name: str = Field(description="Name of the dish refined and translated to ukrainian.")
    type: str = Field(description="Type of the dish refined and translated to ukrainian.")
    ingredients: str = Field(description="Ingredients of the dish from the retrieved data.")
    description: str = Field(
        description="Steps to prepare the dish from the retrieved data. You must always rephrase and enrich it yourself"
    )
    comments: str = Field(
        description="You must always add your personal thoughts on recipe here."
    )

def format_docs(docs):
    return "\n\n".join(
        [f"id: {doc.metadata['id']}\n" + doc.page_content for doc in docs]
    )

def build_recipe_rag():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
    )

    dish_retriever = get_vector_db("dishes").as_retriever(search_kwargs={"k": 4})

    
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    parser = PydanticToolsParser(tools=[Dish])

    chain = (

        {   
            "input": lambda x: x["input"],
            "context": (lambda x: x["input"]) | dish_retriever | format_docs,
        }
        | prompt
        | llm.bind_tools([Dish])
        | parser
    )
    return chain

if __name__ == "__main__":
    rag_chain = build_recipe_rag()
    res = rag_chain.invoke(input={"input": "дай рецепт млинців"})
    print(res)
