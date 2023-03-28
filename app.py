import streamlit as st
import time
from PIL import Image
st.title('FashionGPT')

style_names = ['face', 'hair', 'headwear', 'background', 'top', 'outer', 'bottom', 'shoes', 'accesories']

left_column, right_column = st.columns([1,1])

# with right_column:
#     f_path = '/home/soon/datasets/deepfashion_multimodal/images/WOMEN-Tees_Tanks-id_00007976-01_4_full.jpg'
#     image = Image.open(f_path)
#     st.image(f_path, width=512)
gen_image = right_column.empty()

with left_column:
    with st.form(key='input'):    
        context_text = st.text_area('Content text')
        st.write("Style Texts")
        
        style_columns = st.columns(3)

        style_texts = []
        for i, style in enumerate(style_names):
            col = i//3
            with style_columns[col]:
                style_texts.append(st.text_input(style))

        st.markdown("---")
        submit_button = st.form_submit_button(label='Submit')
        #submit_button = st.form_submit_button("Generate")
        if submit_button:
            #st.write('Content text:', style_texts)
            f_path = '/home/soon/datasets/deepfashion_multimodal/images/WOMEN-Tees_Tanks-id_00007976-01_4_full.jpg'
            image = Image.open(f_path)
            gen_image.image(f_path, width=512)


left_2_column, right_2_column = st.columns([1,1])
style_image = right_2_column.empty()

with left_2_column:
    st.write("Style References")
    style_file = st.file_uploader("Style reference")
    if style_file:
        st.write("lala", style_file.name)
        bytes_data = style_file.read()
        style_image.image(bytes_data)
        options = st.multiselect('Select styles', style_names, [])

