## 1. Dataset y preparación

- [x] 1.1 Implementar la carga del SMS Spam Collection de UCI y documentar fuente, atribución, campos y etiquetas `ham`/`spam`; validar formato, etiquetas inesperadas y filas vacías.
- [x] 1.2 Implementar limpieza conservadora reproducible y análisis exploratorio de distribución de clases, longitudes y ejemplos de mensajes; calcular y presentar los valores al ejecutar, sin inventarlos.
- [x] 1.3 Crear particiones estratificadas aproximadas 80/10/10 para entrenamiento, validación y prueba, con semilla fija documentada; comprobar proporciones y mantener prueba fuera del ajuste y selección.

## 2. Entrenamiento y evaluación

- [x] 2.1 Crear el código de entrenamiento TensorFlow/Keras con `TextVectorization` adaptada solo en entrenamiento, embedding con máscara, capa GRU y salida sigmoide; documentar el mapeo de clase y umbral base 0.5.
- [ ] 2.2 Entrenar y seleccionar configuración usando entrenamiento/validación, registrar historial y evaluar el conjunto de prueba una sola vez al final.
- [x] 2.3 Calcular con predicciones reales precisión, recall, F1, ROC-AUC, PR-AUC cuando esté disponible, reporte por clase y matriz de confusión; generar gráficas de entrenamiento y evaluación.
- [x] 2.4 Incluir en el notebook una tabla con al menos diez predicciones de prueba y etiquetas reales (o todas si la partición tiene menos de diez), sin resultados precargados ficticios.
- [ ] 2.5 Guardar el modelo completo en `models/sms_spam_gru.keras`, volver a cargarlo en otro proceso y confirmar que recibe texto crudo y contiene la preparación/vectorización aprendida.

## 3. Notebook, predicción y aplicación

- [x] 3.1 Crear el notebook `.ipynb` ejecutable en Google Colab que describa el dataset y presente carga, análisis exploratorio, limpieza, particiones, entrenamiento, métricas, gráficas, matriz de confusión y predicciones; conectarlo con el código reutilizable.
- [x] 3.2 Crear el módulo/código de predicción para cargar el modelo guardado, enviar mensajes sin preprocesamiento duplicado y devolver etiqueta spam/legítimo y confianza.
- [x] 3.3 Crear la app Streamlit para ingresar mensajes nuevos y mostrar predicción y confianza; manejar entradas vacías y modelo ausente o ilegible con instrucciones comprensibles.

## 4. Dependencias y entrega

- [x] 4.1 Añadir `requirements.txt` con TensorFlow/Keras, pandas, scikit-learn, herramientas de gráficas y Streamlit, usando versiones compatibles con la serialización `.keras`.
- [x] 4.2 Documentar en README descripción del dataset, atribución, instalación, ejecución local y en Colab, generación del modelo y lanzamiento de Streamlit; distinguir resultados medidos de resultados pendientes.
- [ ] 4.3 Tras una ejecución real del entrenamiento, incluir `models/sms_spam_gru.keras` en el repositorio y cualquier salida de resultados que se decida conservar, verificando que coincida con las métricas y predicciones generadas; no declarar resultados antes de esa ejecución.
