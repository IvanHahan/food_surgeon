
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, tool
from langchain.chains import RetrievalQA
from langchain_together import ChatTogether

from food_surgeon.rag import get_db

load_dotenv(".env")

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
)

db = get_db('dishes')
retriever = db.as_retriever()
qa_rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@tool
def search_db(query: str) -> str:
    """Search the database for a query."""
    return qa_rag.run(query)


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
    res = db.similarity_search('Cирники', k=1)
    print(res)