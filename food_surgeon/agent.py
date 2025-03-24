import json
import os
import re

from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from food_surgeon.data_model import DishList
from food_surgeon.tools import dish_retriever_tool


# Function to build the recipe agent
def build_recipe_agent(use_togetherai=True):
    """Build the recipe agent."""
    if use_togetherai:
        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        )
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    memory = MemorySaver()

    tools = [dish_retriever_tool]

    if use_togetherai:
        system_message = f"""
            You are a culinary assistant. Respond only in Ukrainian.
            A user may just write a name of a dish, you must search for it in the database.
            If relevant dish not found in database, say that it's not found and ask user for another dish.
            if any dish found, {PydanticOutputParser(pydantic_object=DishList).get_format_instructions()}.
        """
    else:
        system_message = """
            You are a culinary assistant. Respond only in Ukrainian.
            A user may just write a name of a dish, you must search for it in the database.
            If relevant dish is not found, put the structured response empty.
            If relevant dish not found in database, say that it's not found and ask user for another dish.
        """

    # Create the ReAct agent executor
    if use_togetherai:
        langgraph_agent_executor = create_react_agent(
            llm, tools, prompt=system_message, checkpointer=memory
        )
    else:
        langgraph_agent_executor = create_react_agent(
            llm,
            tools,
            prompt=system_message,
            checkpointer=memory,
            response_format=DishList,
        )
    return langgraph_agent_executor


# Must be used with togetherai
def parse(output):
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return None
    try:
        return DishList(**json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return None


# Main function to run the script
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Build and invoke the recipe agent
    agent = build_recipe_agent()
    config = {"configurable": {"thread_id": "test-thread"}}

    res = agent.invoke({"messages": [("user", "Дай рецепт панкейків")]}, config)
    print("\n\nДай рецепт панкейків")
    parsed = parse(res["messages"][-1].content)
    print("Відповідь:", parsed if parsed else res["messages"][-1].content)

    res = agent.invoke(
        {"messages": [("user", "Cкільки в цих панкейках калорій")]}, config
    )
    print("\n\nCкільки в цих панкейках калорій")
    parsed = parse(res["messages"][-1].content)
    print("Відповідь:", parsed if parsed else res["messages"][-1].content)
