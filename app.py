import streamlit as st

from food_surgeon.db import get_firebase_db
from food_surgeon.rag import build_recipe_rag

NUM_IMAGES_PER_ROW = 3

if "rag" not in st.session_state:
    st.session_state.rag = build_recipe_rag()


def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["content"]:
                st.markdown(message["content"])
            if "dish" in message:
                dish_widget(message["dish"])


st.title("Food Surgeon")


def dish_widget(dish):
    with st.expander(dish["name"]):
        st.write(dish["ingredients"])
        st.write(dish["description"])
    if "src" in dish:
        st.image(dish["src"], width=200)
    if "comment" in dish:
        st.markdown(dish["comment"])


if "total_dishes" not in st.session_state:
    dishes = get_firebase_db("dishes").get()
    st.session_state.total_dishes = dishes

types = set(
    [dish["type"] for dish in st.session_state.total_dishes.values() if "type" in dish]
)
tab_titles = list(types)
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

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

if not st.session_state.greetings:
    intro = "Привіт! Я допоможу тобі з готовкою"
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": intro})
    st.session_state.greetings = True

display_chat_messages()

if prompt := (st.chat_input("Що хочеш приготувати?")):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Зачекай, шукаю рецепти..."):
        dishes = st.session_state.rag.invoke(
            {
                "input": st.session_state.messages[-1]["content"],
                # "chat_hisory": st.session_state.messages,
            }
        )
    if isinstance(dishes, list):
        for dish in dishes:
            dish = dish.model_dump()
            message = {"role": "assistant", "content": dish.get("comments"), "dish": dish}
            st.session_state.messages.append(
                message
            )
        st.rerun()
