# Clasificador de SMS spam con GRU

Proyecto académico para clasificar mensajes SMS como spam o legítimos mediante una red GRU con TensorFlow/Keras. Incluye un notebook para Google Colab, entrenamiento reproducible, inferencia de texto crudo y una interfaz Streamlit.

## Dataset

Se utiliza **SMS Spam Collection** de UCI. El corpus contiene mensajes etiquetados `ham` (legítimo) y `spam`, almacenados en un archivo TSV sin encabezado con dos columnas: etiqueta y mensaje. La limpieza elimina mensajes vacíos y recorta espacios exteriores; conserva el texto original para que la normalización y vectorización ocurran dentro del modelo.

La fuente de datos, licencia y atribución son:

> Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5CC84](https://doi.org/10.24432/C5CC84). Licencia CC BY 4.0.

Página del conjunto: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

El entrenamiento descarga el archivo desde UCI a `data/`. Si la descarga no está disponible, descarga `SMSSpamCollection` desde UCI y pásalo con `--data-path`.

## Entrenar localmente

Requiere Python 3.10–3.13. El rango de TensorFlow permite usar las versiones compatibles con cada Python; Python 3.13 resuelve a TensorFlow 2.21.x, mientras que las instalaciones locales con Python 3.10–3.12 mantienen compatibilidad con las versiones publicadas para esos intérpretes.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py --epochs 20 --batch-size 32
```

En macOS/Linux, activa el entorno con `source .venv/bin/activate`. Para usar un archivo descargado manualmente:

```bash
python train.py --data-path data/SMSSpamCollection --epochs 20 --batch-size 32
```

El flujo divide estratificadamente los datos en entrenamiento, validación y prueba (aproximadamente 80/10/10), con semilla 42. La capa `TextVectorization` se adapta solo con entrenamiento y se guarda dentro del modelo. La selección de épocas usa validación; el umbral del reporte base es 0.50. La evaluación final de prueba genera métricas, matriz de confusión, curvas de entrenamiento y hasta diez predicciones reales.

Salidas que se generan al ejecutar el entrenamiento:

- `models/sms_spam_gru.keras`: modelo completo de texto crudo a probabilidad de spam.
- `reports/dataset_exploration.png`: exploración de clases y longitudes.
- `reports/training_curves.png` y `reports/confusion_matrix.png`.
- `reports/metrics.json`, `reports/run_info.json` y `reports/test_predictions.csv`.

No hay métricas ni predicciones publicadas de antemano. Las cifras se producen al ejecutar el entrenamiento con el corpus.

## Ejecutar el notebook en Google Colab

1. Abre `sms_spam_gru_colab.ipynb` en Colab y ejecuta las celdas en orden.
2. Cuando aparezca el selector, sube `train.py` y `requirements.txt` desde este proyecto.
3. La celda siguiente instala las dependencias y el notebook carga el dataset de UCI, explora los datos, divide y entrena el modelo.
4. Si el runtime no puede acceder a UCI, descarga desde la página oficial el archivo `SMSSpamCollection` y súbelo cuando el notebook lo solicite.
5. Al terminar, ejecuta las celdas de guardado. Colab solicitará montar Google Drive y copiará el modelo a `MyDrive/rnn-sms-spam/models/sms_spam_gru.keras` y todos los archivos reales de `reports/` a `MyDrive/rnn-sms-spam/reports/`. Las carpetas se crean si no existen; no se genera un ZIP ni se descargan resultados ficticios.

## App Streamlit

Con el modelo guardado en `models/sms_spam_gru.keras` y el entorno activado:

```bash
streamlit run app.py
```

La app acepta un SMS sin procesar y muestra la clase estimada y la confianza asociada a esa clase. No replica ni altera la preparación de texto; el modelo aplica la misma normalización y vectorización aprendida durante el entrenamiento. La confianza es la salida del clasificador y no implica calibración probabilística. Si el modelo aún no existe, la app indica cómo generarlo.

## Archivos principales

- `train.py`: carga, limpieza, EDA, partición, entrenamiento, evaluación y guardado del modelo.
- `predict.py`: carga del modelo y predicción de un mensaje nuevo.
- `app.py`: interfaz Streamlit.
- `sms_spam_gru_colab.ipynb`: notebook académico ejecutable en Colab.
- `requirements.txt`: dependencias.
- `models/sms_spam_gru.keras`: artefacto generado tras una ejecución real.
