# Spec Delta

## Purpose

Permite a una persona probar mensajes SMS nuevos mediante una interfaz sencilla que presenta la clase estimada y la confianza asociada.

## ADDED Requirements

### Requirement: Predicción directa de texto nuevo
La aplicación SHALL permitir ingresar un mensaje sin procesar y SHALL usar el modelo guardado para calcular su clase y confianza sin duplicar el preprocesamiento de entrenamiento.

#### Scenario: Predicción de un mensaje válido
- **WHEN** la persona envía un SMS no vacío y el artefacto del modelo está disponible
- **THEN** la aplicación presenta la clase spam o legítimo y la probabilidad correspondiente como confianza

#### Scenario: Entrada vacía
- **WHEN** la persona intenta predecir una entrada vacía o compuesta solo por espacios
- **THEN** la aplicación solicita un mensaje antes de mostrar una clasificación

### Requirement: Errores de disponibilidad del modelo comprensibles
La aplicación SHALL informar de forma clara cuando no pueda cargar el modelo requerido para predecir.

#### Scenario: Modelo ausente o ilegible
- **WHEN** la aplicación inicia sin un artefacto cargable en la ruta configurada
- **THEN** muestra una instrucción accionable para generar o ubicar el modelo y no presenta una clase ficticia
