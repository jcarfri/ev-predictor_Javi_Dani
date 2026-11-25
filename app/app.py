import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Predictor de compra de vehículo eléctrico",
                   page_icon="🔌", layout="wide")


@st.cache_resource
def cargar():
    modelo = joblib.load(BASE / "models" / "modelo_ev.joblib")
    with open(BASE / "models" / "meta_modelo.json", encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, meta


def widget_numerico(col, minimo, maximo):
    es_entero = float(minimo).is_integer() and float(maximo).is_integer()
    if es_entero and (maximo - minimo) <= 10:          # 0/1, nº de vehículos...
        return st.selectbox(col, list(range(int(minimo), int(maximo) + 1)))
    if es_entero:
        return st.number_input(col, min_value=int(minimo), max_value=int(maximo),
                               value=int((minimo + maximo) // 2), step=1)
    return st.number_input(col, min_value=float(minimo), max_value=float(maximo),
                           value=float((minimo + maximo) / 2))


modelo, meta = cargar()
umbral = meta["umbral"]

st.title("🔌 ¿Comprará un vehículo eléctrico?")
st.caption("Modelo de clasificación entrenado con el dataset 'Predicting Electric Vehicle Purchase' (Kaggle).")

with st.form("formulario"):
    st.subheader("Perfil de la persona")
    columnas = st.columns(3)
    valores = {}
    for i, col in enumerate(meta["features"]):
        with columnas[i % 3]:
            if col in meta["cat_cols"]:
                valores[col] = st.selectbox(col, meta["categorias"][col])
            else:
                minimo, maximo = meta["rangos_num"][col]
                valores[col] = widget_numerico(col, minimo, maximo)
    enviar = st.form_submit_button("Predecir", type="primary")

if enviar:
    entrada = pd.DataFrame([valores])[meta["features"]]
    puntuacion = float(modelo.predict_proba(entrada)[0, 1])
    compra = puntuacion >= umbral

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Puntuación del modelo", f"{puntuacion:.1%}")
    c2.metric("Umbral de decisión", f"{umbral:.1%}")
    st.progress(min(puntuacion, 1.0))

    if compra:
        st.success("✅ El modelo predice que **SÍ comprará** un vehículo eléctrico.")
    else:
        st.info("❌ El modelo predice que **NO comprará** un vehículo eléctrico.")

    st.caption("La puntuación no es una probabilidad calibrada (el modelo se entrenó "
               "con pesos de clase balanceados). Úsala para ordenar perfiles, no como "
               "probabilidad real de compra.")