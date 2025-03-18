
import uuid

import streamlit as st
from dotenv import load_dotenv

from food_surgeon.agent import build_recipe_agent, parse
from food_surgeon.db import get_firebase_db

NUM_IMAGES_PER_ROW = 3
USE_TOGETHERAI = True

load_dotenv()

def initialize_session_state():
    """Initialize session state variables if they are not already set."""
    if "rag" not in st.session_state:
        st.session_state.rag = build_recipe_agent(use_togetherai=USE_TOGETHERAI)
    if "total_dishes" not in st.session_state:
        dishes = get_firebase_db("dishes").get()
        st.session_state.total_dishes = dishes
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.greetings = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def display_chat_messages() -> None:
    """Display chat messages from the session state."""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["content"]:
                st.markdown(message["content"])
            if "dish" in message:
                dish_widget(message["dish"])

def dish_widget(dish):
    """Display a dish widget with its details.
    
    Args:
        dish (dict): A dictionary containing dish details.
    """
    with st.expander(dish["name"]):
        st.write(dish["ingredients"])
        st.write(dish["description"])
    if "src" in dish:
        st.image(dish["src"], width=200)
    if "comment" in dish:
        st.markdown(dish["comment"])

def display_dishes_by_type():
    """Display dishes categorized by their type in separate tabs."""
    types = set(
        [dish["type"] for dish in st.session_state.total_dishes.values() if "type" in dish]
    )
    tab_titles = sorted(list(types))
    tabs = st.tabs(tab_titles)
    for tab, dish_type in zip(tabs, tab_titles):
        with tab:
            cols = st.columns(NUM_IMAGES_PER_ROW)
            dish_list = [
                dish
                for dish in st.session_state.total_dishes.values()
                if dish.get("type") == dish_type
            ]
            for i, dish in enumerate(dish_list):
                with cols[i % NUM_IMAGES_PER_ROW]:
                    dish_widget(dish)

def handle_user_input():
    """Handle user input and update chat messages accordingly."""
    if not st.session_state.greetings:
        intro = "Привіт! Я допоможу тобі з готовкою"
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

    display_chat_messages()

    if prompt := (st.chat_input("Що хочеш приготувати?")):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Зачекай, шукаю рецепти..."):
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            response = st.session_state.rag.invoke(
                {
                    "messages": ("user", st.session_state.messages[-1]["content"]),
                },
                config
            )
            if USE_TOGETHERAI:
                dish_list = parse(response["messages"][-1].content)
            else:
                dish_list = response["structured_response"]
        if not dish_list:
            st.session_state.messages.append(
                    {"role": "assistant", "content": response["messages"][-1].content}
                )
        elif not dish_list.dishes:
            if USE_TOGETHERAI:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "На жаль таку страву я не зміг знайти :("}
                ) 
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": response['messages'][-1].content}
                )
        else:
            for dish in dish_list.dishes:
                dish = dish.model_dump()
                st.session_state.messages.append(
                    {"role": "assistant", "content": dish.get("comments"), "dish": dish}
                )
        st.rerun()

# Initialize session state
initialize_session_state()

# Set the title of the app
st.title("Food Surgeon")

# Display dishes by type
display_dishes_by_type()

# Add a divider
st.divider()

# Handle user input and chat messages
handle_user_input()
