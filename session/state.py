import streamlit as st


def init_session_states(*states):
    for state in states:
        if state not in st.session_state:
            st.session_state[state] = 0


def increment_counter(counter_state):
    st.session_state[counter_state] += 1
