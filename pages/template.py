import streamlit as st


def section_1():
    st.write("Hello!")


def section_2():
    st.write(2)


page_names_to_funcs = {
    "Section 1": section_1,
    "Section 2": section_2,
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())

page_names_to_funcs[selected_page]()
