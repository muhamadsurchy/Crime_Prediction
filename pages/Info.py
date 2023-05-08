import streamlit as st
from PIL import Image

st.markdown("<h3 style='text-align: center; color: black;'>Information of DataSet</h3>", unsafe_allow_html=True)

image = Image.open('dataset.jpg')
st.image(image)

st.markdown("<pre style='text-align: center; color: black;'>Your DataSet File Must Be .CSV Format</pre>", unsafe_allow_html=True)
st.markdown("<pre style='text-align: center; color: black;'>If You Want To Use This Model To Your Company, You Have To Call Us To Customize The Model With Your Company. Otherwise It Will Not Working!</pre>", unsafe_allow_html=True)

#------------------------------------------------------------------

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#------------------------------------------------------------------