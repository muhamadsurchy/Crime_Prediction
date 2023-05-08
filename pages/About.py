import streamlit as st
from PIL import Image
from pathlib import Path

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "style.css"

# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center; color: black;'>About The Developers Of The Model</h3>", unsafe_allow_html=True)

#opening the image
imagek = Image.open('Khayam.jpg')
new_khayam = imagek.resize((600, 600))
images = Image.open('Sarhad.jpg')
new_sarhad = imagek.resize((600, 800))
imagem = Image.open('Muhamad.jpg')
#displaying the image on streamlit app

kplatform1 = 'Telegram'
klink1 = 'https://t.me/KhayyamSherzad'
kplatform2 = 'Facebook'
klink2 = 'https://www.facebook.com/khayyam.sherzad?mibextid=LQQJ4d'

splatform1 = 'Telegram'
slink1 = 'https://t.me/MuhamadSurchy'
splatform2 = 'Facebook'
slink2 = 'https://www.facebook.com/sarhad.Baez?mibextid=LQQJ4d'

mplatform1 = 'Telegram'
mlink1 = 'https://t.me/MuhamadSurchy'
mplatform2 = 'Facebook'
mlink2 = 'https://www.facebook.com/profile.php?id=100055796531828&mibextid=LQQJ4d'

col1, col2, col3 = st.columns(3)

with col1:
   st.image(new_khayam, caption='Khayam Sherzad \ Developer')
   # --- SOCIAL LINKS ---
   f"[{kplatform1}]({klink1})"
   f"[{kplatform2}]({klink2})"
   


with col2:
   st.image(images, caption='Sarhad Baez \ Supervisor')
   # --- SOCIAL LINKS ---
   f"[{splatform1}]({slink1})"
   f"[{splatform2}]({slink2})"

with col3:
   st.image(imagem, caption='Muhamad Mahmud \ Developer')
   # --- SOCIAL LINKS ---
   f"[{mplatform1}]({mlink1})"
   f"[{mplatform2}]({mlink2})"

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