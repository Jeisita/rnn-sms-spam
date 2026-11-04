# Design

## Context

El repositorio parte sin código de aplicación ni especificaciones previas. Ver [proposal.md](proposal.md) para la motivación y alcance; las capacidades están definidas en `specs/sms-spam-classification/spec.md` y `specs/sms-spam-inference-app/spec.md`. La fuente acordada es SMS Spam Collection de UCI y el entorno docente objetivo incluye Google Colab.

## Goals / Non-Goals

**Goals:**

- Hacer que entrenamiento, notebook y aplicación compartan un modelo de texto autocontenido.
- Mantener separada la prueba del ajuste y de la selección de umbral.
- Permitir repetir el flujo desde datos descargados hasta artefactos documentados.

**Non-Goals:**

- Desplegar un servicio en producción o procesar mensajes personales de forma persistente.
- Afirmar que la confianza sea una probabilidad calibrada sin evaluar calibración.
- Garantizar un resultado mínimo de exactitud o rendimiento antes de ejecutar el experimento.

## Decisions

1. **Usar un único módulo de entrenamiento invocable desde Colab y localmente.** El notebook explicará las etapas académicas y llamará funciones reutilizables para evitar que una segunda implementación se desvíe del script. Alternativa considerada: implementar toda la lógica solo en celdas; se descarta porque dificulta repetir entrenamiento y evaluación desde la aplicación de desarrollo.

2. **Obtener el corpus UCI y validar su esquema antes de continuar.** El flujo documentará la fuente y su atribución, comprobará columnas y etiquetas esperadas, y fallará con un mensaje claro si descarga o lectura no funciona. El procesamiento usará etiquetas `ham` y `spam`, mapeadas a legítimo y spam. Alternativa considerada: incluir una copia de datos sin procedencia; se descarta para conservar trazabilidad.

3. **Dividir con dos cortes estratificados y semilla fija documentada.** Se reservará primero un conjunto de prueba y luego se separará validación del resto de desarrollo, de modo que las proporciones objetivo sean aproximadamente 80/10/10. Se comprobarán las proporciones y la presencia de ambas clases. La partición de prueba solo se consultará al final. Alternativa considerada: dividir sin estratificación o evaluar repetidamente sobre prueba; se descarta por riesgo de desbalance o contaminación.

4. **Usar `TextVectorization` dentro del grafo Keras.** La capa se adaptará exclusivamente con mensajes de entrenamiento, mantendrá vocabulario y normalización de minúsculas/puntuación de forma consistente y se conectará a embeddings con máscara, GRU y salida sigmoide. El `.keras` guardado recibirá texto crudo y devolverá la probabilidad de spam. Alternativa considerada: limpiar/vectorizar fuera del modelo o en la app; se descarta porque crea dos rutas de preprocesamiento.

5. **Fijar el umbral de clasificación en 0.5 para el reporte base.** Se informarán precisión, recall, F1, matriz de confusión y ROC-AUC, además de curvas/métricas de entrenamiento; se incluirá PR-AUC cuando esté disponible en la evaluación. El umbral y mapeo de clases quedarán documentados. Alternativa considerada: escoger umbral con el conjunto de prueba; se descarta por fuga de información. Cualquier optimización posterior debe usar validación y quedar claramente identificada.

6. **Guardar artefactos con rutas estables y desacoplar interfaz de modelo.** El entrenamiento escribirá el modelo en `models/sms_spam_gru.keras`; un módulo de predicción cargará ese artefacto, y Streamlit delegará en ese módulo para mostrar clase y confianza. Si el archivo no está, la interfaz indicará cómo generarlo. El notebook generará tablas y figuras desde la ejecución real, incluidas diez predicciones de prueba o todas las disponibles si hay menos de diez.

7. **Mantener el notebook como material reproducible, no como reporte con resultados ficticios.** El README describirá instalación, descarga, ejecución y ubicación del modelo. No habrá métricas ni predicciones estáticas inventadas; si se entrega un modelo en el repositorio, debe proceder de una ejecución completa documentada y sus resultados serán los mismos que produjo esa ejecución.

## Risks / Trade-offs

- [Descarga de UCI no disponible en Colab o local] → Documentar fuente, formato esperado y un procedimiento alternativo reproducible para colocar el archivo descargado; validar antes del análisis.
- [Variaciones entre versiones de TensorFlow/Keras al serializar capas] → Fijar rangos de dependencias compatibles, guardar en formato `.keras` y documentar una prueba de carga e inferencia.
- [Desbalance de clases puede ocultar fallos en accuracy] → Reportar precision, recall, F1, matriz de confusión y métricas de ranking; explicar falsos positivos y falsos negativos.
- [Artefacto del modelo aumenta el tamaño del repositorio] → Revisar el tamaño generado y documentar cómo regenerarlo; mantener el archivo solicitado en la ruta acordada cuando la ejecución se haya realizado.
- [Confianza puede interpretarse como certeza] → Presentarla como probabilidad de salida del clasificador, con nota de que no implica calibración.
- [La variabilidad de entrenamiento puede cambiar las métricas] → Fijar semillas cuando sea posible, registrar versiones y no prometer determinismo absoluto entre hardware o versiones.

## Migration Plan

No hay migración de usuarios ni datos existentes. Para una entrega reproducible: instalar dependencias, descargar/leer la fuente UCI, ejecutar el notebook o script, revisar las métricas y predicciones realmente generadas, guardar `models/sms_spam_gru.keras` y luego iniciar Streamlit. Para revertir, retirar los nuevos archivos de proyecto; no hay cambios en sistemas desplegados.

