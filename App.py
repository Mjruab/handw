import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Digit AI Studio",
    page_icon="✍️",
    layout="wide"
)

# ─────────────────────────────────────────────
# ESTILOS (MISMO TEMA)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #fffde7; color: #333333; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #fff9c4 !important;
    border-right: 1px solid #f9a825;
}

/* Títulos */
h1 { color: #f57f17 !important; font-weight: 700 !important; }
h2, h3 { color: #e65100 !important; }

/* Botón */
.stButton > button {
    background: #f9a825 !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #f57f17 !important;
}

/* Cards */
.header-card {
    background: #fff8e1;
    border-left: 5px solid #f9a825;
    padding: 25px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.section-card {
    background: #fff8e1;
    border: 1px solid #ffe082;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA DE MODELO (OPTIMIZADO)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

model = load_model()

# ─────────────────────────────────────────────
# FUNCIÓN DE PREDICCIÓN
# ─────────────────────────────────────────────
def predictDigit(image):
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32') / 255
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    return np.argmax(pred[0])

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1 style="margin:0;">✍️ Digit AI Studio</h1>
    <p style="margin:5px 0 0 0; color:#f57f17;">
        Reconocimiento de dígitos escritos a mano con redes neuronales
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────
col1, col2 = st.columns([1,1], gap="large")

# ─────────────────────────────────────────────
# CANVAS
# ─────────────────────────────────────────────
with col1:
    
    st.markdown("### ✏️ Dibuja tu número")

    stroke_width = st.slider('Grosor del trazo', 1, 30, 15)

    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=250,
        width=250,
        drawing_mode="freedraw",
        key="canvas",
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESULTADO
# ─────────────────────────────────────────────
with col2:
    
    st.markdown("### 🤖 Predicción")

    if st.button("PREDECIR ↗"):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')

            resultado = predictDigit(input_image)

            st.success(f"🎯 El dígito es: **{resultado}**")
        else:
            st.warning("⚠️ Dibuja un número primero")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ Digit AI Studio")
    st.divider()

    st.markdown("### Acerca de")
    st.markdown("""
    Esta aplicación utiliza una **Red Neuronal Artificial (RNA)** entrenada
    para reconocer dígitos escritos a mano.

    🧠 Modelo basado en Deep Learning  
    ✏️ Entrada mediante dibujo en canvas  
    ⚡ Predicción en tiempo real  
    """)

    st.divider()

    st.markdown("### Cómo usar")
    st.markdown("""
    1. Dibuja un número (0–9)  
    2. Ajusta el grosor si quieres  
    3. Presiona **PREDECIR**  
    """)





