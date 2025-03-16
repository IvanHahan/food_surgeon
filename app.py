import streamlit as st

from food_surgeon.db import get_firebase_db
from food_surgeon.rag import run_agent

NUM_IMAGES_PER_ROW = 3


def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        # with st.sidebar:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message:
                for i in range(0, len(message["images"]), NUM_IMAGES_PER_ROW):
                    cols = st.columns(NUM_IMAGES_PER_ROW)
                    for j in range(NUM_IMAGES_PER_ROW):
                        if i + j < len(message["images"]):
                            cols[j].image(message["images"][i + j], width=200)

st.title("Food Surgeon")


def dish_widget(dish, key=None):
    with st.container():
        if st.button(dish['name'], key=key):
            st.session_state['selected_dish'] = dish
            st.rerun()
        if 'src' in dish:
            st.image(dish['src'], width=200)

def dish_widget_detailed(dish):
    with st.container():
        if st.button('Back'):
            st.session_state.pop('selected_dish', None)
            st.rerun()
        st.subheader(dish['name'])
        st.write(f"Тип: {dish.get('type')}")
        st.write(f"Інгредієнти: {dish.get('ingredients')}")
        st.write(f"Опис: {dish.get('description')}")
        if 'src' in dish:
            st.image(dish['src'], width=200)


if 'selected_dish' in st.session_state:
    dish_widget_detailed(st.session_state['selected_dish'])
else:
    dishes = get_firebase_db('dishes').get()
    types = set([dish['type'] for dish in dishes.values() if 'type' in dish])
    tab_titles = list(types)
    tabs = st.tabs(tab_titles)
    for tab, dish_type in zip(tabs, tab_titles):
        with tab:
            cols = st.columns(NUM_IMAGES_PER_ROW)
            dish_list = [dish for dish in dishes.values() if dish.get('type') == dish_type]
            for i, dish in enumerate(dish_list):
                with cols[i % NUM_IMAGES_PER_ROW]:
                    dish_widget(dish)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False


display_chat_messages()

if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "Привіт! Я допоможу тобі з готовкою"
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True    


if prompt := (st.chat_input("Що хочеш приготувати?")):
# Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    dishes = run_agent(st.session_state.messages)
    if isinstance(dishes, list):
        for dish in dishes:
            dish_widget(dish, key=dish['id'])
    elif isinstance(dishes, str):
        with st.chat_message("assistant"):
            st.markdown(dishes)
 
        # dish_widget(response)
