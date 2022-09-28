import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


@st.experimental_memo()
def random_plot(counter):
    x = np.linspace(0, 100, 100)
    y = np.random.randn(100).cumsum()
    c = np.random.choice(['b', 'r', 'g'])
    fig, ax = plt.subplots()
    ax.plot(x, y, c=c)
    return fig


def init_session_state(*states):
    for state in states:
        if state not in st.session_state:
            st.session_state[state] = 0


def increment_counter(counter_state):
    st.session_state[counter_state] += 1


def section_1():
    init_session_state('counter_1', 'counter_2')

    st.markdown("# How to reload a single plot")
    p_1 = st.empty()

    counter_1 = st.session_state['counter_1']
    p_1.pyplot(random_plot(counter_1))

    st.button("Recharger plot 1", on_click=increment_counter,
              args=('counter_1',))

    p_2 = st.empty()

    counter_2 = st.session_state['counter_2']
    p_2.pyplot(random_plot(counter_2))

    st.button("Recharger plot 2", on_click=increment_counter,
              args=('counter_2',))


def section_2():
    st.write(2)


page_names_to_funcs = {
    "Section 1": section_1,
    "Section 2": section_2,
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
