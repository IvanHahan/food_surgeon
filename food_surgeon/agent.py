from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, tool
from langchain_together import ChatTogether

from food_surgeon.rag import get_firebase_db, get_vector_db

load_dotenv(".env")

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
)

vector_db = get_vector_db('dishes')

@tool(return_direct=True)
def search_db(query: str) -> str:
    """Search the database for a query."""
    doc = vector_db.similarity_search(query, k=1)[0]
    id_ = doc.metadata['id']
    # Assuming you have a Firebase client initialized as `firebase_db`
    item = get_firebase_db('dishes').child(id_).get()
    return item


# Define any tools you want the agent to use
tools = [search_db]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_agent(prompt: str) -> str:
    return agent.run(prompt)


if __name__ == "__main__":
    # Initialize RAG
    res = run_agent("дай рецепт сирників")

    print(res)