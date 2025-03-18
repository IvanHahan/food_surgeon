
from langchain import hub
from langchain.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI

from food_surgeon.db import get_vector_db
from food_surgeon.tools import format_docs


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

# Main function to run the script
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Build and invoke the RAG chain
    rag_chain = build_recipe_rag()
    res = rag_chain.invoke(input={"input": "дай рецепт млинців"})
    print(res)