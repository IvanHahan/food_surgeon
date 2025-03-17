import streamlit as st

from food_surgeon.db import get_firebase_db
from food_surgeon.rag import build_recipe_rag

NUM_IMAGES_PER_ROW = 3

recipe_rag = build_recipe_rag()

def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message['content']:
                st.markdown(message["content"])
            if "dish" in message:
                dish_widget(message['dish'])

st.title("Food Surgeon")


def dish_widget(dish, key=None):
    # with st.container(key=key):
    with st.expander(dish['name']):
        st.write(dish['ingredients'])
        st.write(dish['description'])
    if 'src' in dish:
        st.image(dish['src'], width=200)
    if 'comment' in dish:
        st.markdown(dish['comment'])

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
    with st.spinner('Зачекай, шукаю рецепти...'):
        dishes = recipe_rag.invoke({'input': st.session_state.messages[-1]['content']})
    if isinstance(dishes, list):
        for dish in dishes:
            dish = dish.model_dump()
            st.session_state.messages.append({"role": "assistant", "content": dish.get('comments'), 'dish': dish})
        st.rerun()