from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers import PydanticToolsParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from food_surgeon.db import get_vector_db

load_dotenv(".env")


class Dish(BaseModel):
    """Always use this tool to structure your response to the user."""

    id: str = Field(description="The id of the dish.")
    name: str = Field(description="The name of the dish.")
    type: str = Field(description="The type of the dish.")
    ingredients: str = Field(description="The ingredients of the dish.")
    description: str = Field(
        description="The description of the dish or steps to prepare"
    )
    comments: str = Field(description="The models's comments on the dish.")

def format_docs(docs):
    return "\n\n".join(
        [f"id: {doc.metadata['id']}\n" + doc.page_content for doc in docs]
    )

def build_recipe_rag():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
    )

    dish_retriever = get_vector_db("dishes").as_retriever()

    retriever_tool = create_retriever_tool(
        dish_retriever,
        "recipe-seeker",
        "Use to search recipe in database",
    )

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
    # Initialize RAG
    # res = run_agent(
    #     [
    #         {"role": "user", "content": "дай рецепт борщу"},
    #     ]
    # )
    res = chain.invoke(input={"input": "дай рецепт борщу"})
    # ("дай рецепт сирників")
    # print(dish_rag.invoke('борщ', k=))
    print(res)
