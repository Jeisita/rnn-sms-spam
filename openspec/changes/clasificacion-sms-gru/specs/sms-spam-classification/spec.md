# Spec Delta

## Purpose

Define un flujo académico reproducible para clasificar SMS como spam o legítimos, evaluar el modelo con datos no vistos y entregar un artefacto listo para inferencia.

## ADDED Requirements

### Requirement: Dataset y análisis exploratorio documentados
El proyecto SHALL identificar la fuente UCI SMS Spam Collection, describir sus campos y clases, y presentar análisis exploratorio reproducible de las etiquetas y los mensajes.

#### Scenario: Inspección del conjunto de datos
- **WHEN** se ejecuta el flujo de análisis con el dataset disponible
- **THEN** se informa el número de ejemplos, la distribución de clases y observaciones descriptivas de los textos sin atribuir cifras que no provengan de esa ejecución

### Requirement: Particiones estratificadas sin contaminación
El flujo SHALL separar ejemplos en entrenamiento, validación y prueba preservando la proporción de clases, y reservar la prueba para la evaluación final.

#### Scenario: Creación de particiones
- **WHEN** el conjunto de datos se divide con la semilla documentada
- **THEN** cada partición contiene ambas clases y los ejemplos de una partición no se usan para ajustar el modelo con otra

### Requirement: Evaluación y reporte reproducibles
El flujo SHALL entrenar una red GRU y reportar métricas adecuadas para clases desbalanceadas, gráficos de entrenamiento, matriz de confusión y al menos diez predicciones de ejemplos de prueba con sus etiquetas reales y predichas.

#### Scenario: Ejecución de evaluación
- **WHEN** termina el entrenamiento y evaluación con éxito
- **THEN** se generan las métricas y visualizaciones desde las predicciones reales de prueba, incluidos diez ejemplos como mínimo cuando la partición de prueba tenga al menos diez filas

#### Scenario: Partición de prueba pequeña
- **WHEN** la partición de prueba contiene menos de diez filas
- **THEN** se muestran todas las predicciones disponibles y se informa que el tamaño de la partición limita el mínimo solicitado

### Requirement: Modelo autocontenido para inferencia
El entrenamiento SHALL guardar un modelo que incluya la preparación y vectorización de texto necesarias para recibir SMS sin procesar y producir probabilidades de clase.

#### Scenario: Guardado y carga del modelo
- **WHEN** el entrenamiento finaliza correctamente y el artefacto se carga en otro proceso
- **THEN** el modelo acepta directamente texto nuevo y devuelve una probabilidad de spam usando la misma vectorización aprendida durante el entrenamiento

### Requirement: Resultados sin cifras ficticias
La documentación y el notebook SHALL distinguir resultados medidos de resultados pendientes y no SHALL presentar métricas ni predicciones como observadas antes de una ejecución real.

#### Scenario: Proyecto aún no ejecutado
- **WHEN** se revisa el repositorio antes del primer entrenamiento reproducible
- **THEN** se describen métricas y predicciones esperadas como salidas por generar, sin inventar sus valores
