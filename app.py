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
    matrix_container.write("Nhập các phần tử của ma trận")
    st.session_state['matrix'] = matrix_container.data_editor(
        data=np.zeros((st.session_state['rows'], st.session_state['cols'])),
        use_container_width=True,
        hide_index=True,
    )

st.title('Phân Rã Ma Trận')
st.write("""**Chú ý**: _nicematrix_ không tương thích. (Không có highlighting)""")
st.write("""**Chú ý**: Đây là một sản phẩm khả thi tối thiểu (MVP).""")
st.write('Truy cập Github để có thêm thông tin.')
st.divider()

#
row_input, col_input = st.columns(2)
with row_input:
    st.session_state['rows'] = st.number_input(
        "Nhập số hàng",
        2,
        6,
        "min",
        1,
        "%d",
        label_visibility="visible",
    )

with col_input:
    st.session_state['cols'] = st.number_input(
        "Nhập số cột",
        2,
        6,
        "min",
        1,
        "%d",
        label_visibility="visible",
    )

update_matrix()

decompose_button = st.button("Phân Rã Ma Trận")
if decompose_button:
    # Streamlit's input is in ndarray
    matrix = np.asmatrix(st.session_state['matrix'], dtype=np.float_)

    with st.spinner('Đang xử lý...'):
        write_pdf(matrix)

    with open("decomposition.pdf", "rb") as f:
        byte = f.read()

    st.download_button("Download PDF", data=byte, file_name="decomposition.pdf", mime="application/octet-stream")

    del st.session_state['rows']
    del st.session_state['cols']
    del st.session_state['matrix']
