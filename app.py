import streamlit as st

from food_surgeon.agent import run_agent
from food_surgeon.rag import get_firebase_db


def dish_widget(dish):
    with st.container():
        st.subheader(dish['name'])
        if 'src' in dish:
            st.image(dish['src'])
        else:
            st.empty()
        if 'type' in dish:
            st.write(f"Type: {dish['type']}")
        st.write(f"Опис: {dish['description']}")
        st.write(f"Інгредієнти: {dish['ingredients']}")

if 'response' in st.session_state:
    dish_widget(st.session_state['response'])
    if st.button("Reset"):
        st.session_state.pop('response', None)
        st.rerun()
else:
    dishes = get_firebase_db('dishes').get()
    types = set([dish['type'] for dish in dishes.values() if 'type' in dish])
    cols = st.columns(min(5, len(types)))
    for i, dish_type in enumerate(types):
        col = cols[i % 5]
        if col.button(dish_type):
            for dish in dishes.values():
                if dish.get('type') == dish_type:
                    dish_widget(dish)

with st.sidebar:
    st.title("Food Surgeon Chatbot")

    prompt = st.text_input("Enter your query:")

    if st.button("Submit"):
        response = run_agent(prompt)
        st.session_state['response'] = response
        st.rerun()
        # dish_widget(response)
