
from langchain.tools import tool

from food_surgeon.db import get_vector_db


# Function to format documents for display
def format_docs(docs):
    """Format documents for display."""
    return "\n\n".join(
        [f"id: {doc.metadata['id']}\n" + doc.page_content for doc in docs]
    )


@tool
def dish_retriever_tool(input):
    """
    Retrieve dishes recipes from the database.
    """
    dish_retriever = get_vector_db("dishes").as_retriever(search_kwargs={"k": 4})
    docs = dish_retriever.invoke(input)
    return format_docs(docs)