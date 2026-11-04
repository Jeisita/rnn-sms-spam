"""Streamlit UI for the SMS spam GRU model."""

from __future__ import annotations

import streamlit as st

from predict import load_model, predict_message


st.set_page_config(page_title="Clasificador de SMS", page_icon="📩", layout="centered")
st.title("Clasificador de SMS")
st.write("Escribe un mensaje y el modelo estimará si es spam o legítimo.")
st.caption("La confianza corresponde a la salida del clasificador; no implica calibración probabilística.")

message = st.text_area("Mensaje SMS", height=150, placeholder="Escribe o pega aquí el mensaje…")

if st.button("Clasificar mensaje", type="primary"):
    if not message.strip():
        st.warning("Escribe un mensaje antes de clasificarlo.")
    else:
        try:
            model = load_model()
            result = predict_message(message, model=model)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"No se pudo cargar el modelo o clasificar el mensaje: {exc}")
        else:
            if result["class_key"] == "spam":
                st.error(f"Clase predicha: **Spam**")
            else:
                st.success(f"Clase predicha: **Legítimo**")
            st.metric("Confianza de la clase predicha", f"{result['confidence']:.1%}")
            st.progress(result["confidence"])
            st.caption(f"Probabilidad de spam: {result['spam_probability']:.1%} · Umbral: 0.50")
