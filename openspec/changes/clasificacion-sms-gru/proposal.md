# Proposal

## Why

Se necesita un proyecto académico reproducible que permita estudiar la clasificación binaria de SMS y demostrar cómo llevar un modelo de aprendizaje profundo desde el análisis de datos hasta una aplicación de inferencia. El proyecto debe poder ejecutarse en Google Colab y evitar diferencias de preprocesamiento entre entrenamiento y uso interactivo.

## What Changes

- Incorporar un notebook de Colab y un script de entrenamiento para explorar el SMS Spam Collection de UCI, limpiar los mensajes y entrenar una red GRU.
- Documentar el dataset, el análisis exploratorio, la partición estratificada de entrenamiento, validación y prueba, y métricas apropiadas para clases desbalanceadas.
- Generar visualizaciones de entrenamiento/evaluación, matriz de confusión y al menos diez predicciones sobre ejemplos de prueba, calculadas al ejecutar el notebook.
- Guardar un modelo TensorFlow/Keras que integre la preparación y vectorización del texto para que entrenamiento y aplicación compartan exactamente el mismo preprocesamiento.
- Añadir código de predicción, aplicación Streamlit, dependencias y README con instrucciones de ejecución local y en Colab.
- No publicar métricas, ejemplos clasificados ni resultados como hechos hasta que se obtengan mediante una ejecución real del entrenamiento.

## Capabilities

### New Capabilities

- `sms-spam-classification`: análisis y entrenamiento reproducibles de una GRU sobre SMS, evaluación con datos separados y exportación del modelo autocontenido.
- `sms-spam-inference-app`: predicción de mensajes nuevos mediante Streamlit usando el modelo guardado y mostrando clase y confianza.

### Modified Capabilities

Ninguna. El repositorio no contiene especificaciones existentes.

## Impact

- Nuevos artefactos académicos: notebook Jupyter compatible con Colab, módulos Python de entrenamiento y predicción, modelo serializado generado durante la ejecución, aplicación Streamlit, `requirements.txt` y README.
- Dependencias principales: TensorFlow/Keras, pandas, scikit-learn, matplotlib/seaborn y Streamlit.
- Fuente de datos externa: SMS Spam Collection de UCI; el método de descarga y la licencia/atribución se documentarán.
- El modelo binario se producirá ejecutando el entrenamiento. La propuesta no presupone métricas ni asegura rendimiento no medido.
