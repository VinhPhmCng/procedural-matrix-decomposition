import streamlit as st
import numpy as np
from write import write_pdf

if 'rows' not in st.session_state:
    st.session_state['rows'] = 2
if 'cols' not in st.session_state:
    st.session_state['cols'] = 2
if 'matrix' not in st.session_state:
    st.session_state['matrix'] = np.zeros((2, 2))

def update_matrix():
    matrix_container = st.container(border=True)
    matrix_container.st.write("Enter the matrix")
    st.session_state['matrix'] = matrix_container.data_editor(
        data=np.zeros((st.session_state['rows'], st.session_state['cols'])),
        use_container_width=True,
        hide_index=True,
    )

st.title('Procedural Matrix Decomposition')
st.write("""**Note**: _nicematrix_ is incompatible here for some reason.""")
st.write("""**Note**: This is a minimum viable product.""")
st.write('Head to Github for more details.')
st.divider()

#
row_input, col_input = st.columns(2)
with row_input:
    st.session_state['rows'] = st.number_input(
        "Enter the number of rows",
        2,
        6,
        "min",
        1,
        "%d",
        label_visibility="visible",
    )

with col_input:
    st.session_state['cols'] = st.number_input(
        "Enter the number of colums",
        2,
        6,
        "min",
        1,
        "%d",
        label_visibility="visible",
    )

update_matrix()

decompose_button = st.button("Decompose Matrix")
if decompose_button:
    # Streamlit's input is in ndarray
    matrix = np.asmatrix(st.session_state['matrix'], dtype=np.float_)

    with st.spinner('Please wait...'):
        write_pdf(matrix)

    with open("decomposition.pdf", "rb") as f:
        byte = f.read()

    st.download_button("Download PDF", data=byte, file_name="decomposition.pdf", mime="application/octet-stream")

    del st.session_state['rows']
    del st.session_state['cols']
    del st.session_state['matrix']
