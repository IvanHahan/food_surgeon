from langchain import hub
from langchain_openai import ChatOpenAI

from food_surgeon.data_model import DishList
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

    # Define the chain of operations
    chain = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
            "context": (lambda x: x["input"]) | dish_retriever | format_docs,
        }
        | prompt
        | llm.with_structured_output(DishList)
    )
    return chain


# Main function to run the script
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Build and invoke the RAG chain
    rag_chain = build_recipe_rag()
    res = rag_chain.invoke(input={"input": "Дай рецепт пельменів"})
    chat_history = [("user", "дай рецепт пельменів"), ("assistant", str(res))]
    print("\n\nдай рецепт пельменів (нема в базі):", res)

    res = rag_chain.invoke(
        input={"input": "а тепер млинців", "chat_history": chat_history}
    )
    print("\n\nа тепер млинців:", res)
