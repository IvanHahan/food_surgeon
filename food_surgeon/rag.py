from langchain import hub
from langchain.chains import create_history_aware_retriever
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from food_surgeon.db import get_vector_db


# Define the Dish model to structure the response for a single dish
class Dish(BaseModel):
    """Always use this tool to structure your response to the user."""

    id: str = Field(description="Identifier of the dish from the retrieved data.")
    name: str = Field(
        description="Name of the dish refined and translated to ukrainian."
    )
    type: str = Field(
        description="Type of the dish refined and translated to ukrainian."
    )
    ingredients: str = Field(
        description="Ingredients of the dish from the retrieved data."
    )
    description: str = Field(
        description="Steps to prepare the dish from the retrieved data. You must always rephrase and enrich it yourself"
    )
    comments: str = Field(
        description="You must always add your personal thoughts on recipe here."
    )

# Define the DishList model to structure the response when multiple dishes are retrieved
class DishList(BaseModel):
    """Always use this tool to structure your response to the user if you have several dishes as output. Put empty list, if no relevant dish found"""

    dishes: list[Dish] = Field(description="List of dishes.")

# Function to format documents for display
def format_docs(docs):
    """Format documents for display."""
    return "\n\n".join(
        [f"id: {doc.metadata['id']}\n" + doc.page_content for doc in docs]
    )

# Function to build the recipe retrieval-augmented generation (RAG) chain
def build_recipe_rag():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
    )

    # Retrieve dishes from the vector database
    dish_retriever = get_vector_db("dishes").as_retriever(search_kwargs={"k": 4})

    # Pull the prompt from the hub
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=dish_retriever, prompt=prompt
    )

    # Define the chain of operations
    chain = (
        {
            "input": lambda x: x["input"],
            "context": (
                {
                    "chat_history": lambda x: x.get("chat_history"),
                    "input": lambda x: x["input"],
                    "context": lambda x: x.get("context"),
                }
            )
            | history_aware_retriever
            | format_docs,
        }
        | prompt
        | llm.with_structured_output(DishList)
    )
    return chain

# Function to format intermediate steps into a string for the ReAct prompt
def format_agent_scratchpad(intermediate_steps):
    """Format intermediate steps into a string for the ReAct prompt."""
    if not intermediate_steps:
        return ""
    scratchpad = ""
    for action, observation in intermediate_steps:
        scratchpad += f"\nAction: {action.tool}\nInput: {action.tool_input}\nObservation: {observation}\n"
    return scratchpad

# Function to build the recipe agent
def build_recipe_agent():
    """Build the recipe agent."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    dish_retriever = get_vector_db("dishes").as_retriever(search_kwargs={"k": 4})
    memory = MemorySaver()

    @tool
    def dish_retriever_tool(input):
        """
        Retrieve dishes recipes from the database.
        """
        docs = dish_retriever.invoke(input)
        return format_docs(docs)

    tools = [dish_retriever_tool]

    system_message = """
        You are a culinary assistant. Respond only in Ukrainian. 
        If relevant dish is not found, put the structured response empty.
        If relevant dish not found in database, say that it's not found and ask user for another dish.
    """

    # Create the ReAct agent executor
    langgraph_agent_executor = create_react_agent(
        llm, tools, prompt=system_message, response_format=DishList, checkpointer=memory
    )
    return langgraph_agent_executor

# Main function to run the script
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Build and invoke the RAG chain
    rag_chain = build_recipe_rag()
    res = rag_chain.invoke(input={"input": "дай рецепт млинців"})
    print(res)

    # Build and invoke the recipe agent
    agent = build_recipe_agent()
    config = {"configurable": {"thread_id": "test-thread"}}
    res = agent.invoke({"messages": [("user", "людина павук")]}, config)
    print(res["structured_response"])
