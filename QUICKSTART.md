# 🚀 Quick Start Guide

Guía rápida para comenzar con el sistema de recomendación de cultivos.

## Instalación (5 minutos)

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo (1-2 minutos)
```bash
python src/train_model.py
```

Esto generará:
- ✅ Modelo entrenado en `models/crop_model.joblib`
- 📊 Matriz de confusión en `confusion_matrix.png`

## Usar el Sistema

### Opción 1: Interfaz Web (Recomendado)
```bash
streamlit run app.py
```
Luego abre http://localhost:8501 en tu navegador.

### Opción 2: Línea de Comandos
```bash
python src/predict.py
```

### Opción 3: Módulo Python
```python
from src.predict import predict_crop

cultivo = predict_crop(
    N=90,           # Nitrógeno
    P=42,           # Fósforo
    K=43,           # Potasio
    temperature=20.88,   # Temperatura en °C
    humidity=82.00,      # Humedad en %
    ph=6.50,        # pH del suelo
    rainfall=202.94 # Precipitación en mm
)

print(f"Cultivo recomendado: {cultivo}")
```

## Análisis de Datos

Generar gráficos y visualizaciones:
```bash
python src/data_analysis.py
```

Genera:
- 📈 Distribuciones de variables
- 🥧 Distribución de cultivos
- 🔗 Matriz de correlación
- 📊 Análisis por cultivo

## Ejecutar Ejemplo Completo
```bash
python example_usage.py
```

## Estructura del Proyecto

```
crop-recommendation-ml/
├── data/
│   └── Crop_recommendation.csv       # Dataset
├── models/
│   └── crop_model.joblib             # Modelo (se genera)
├── src/
│   ├── train_model.py                # Entrenamiento
│   ├── predict.py                    # Predicciones
│   └── data_analysis.py              # Análisis
├── app.py                            # Aplicación web
├── config.py                         # Configuración
├── example_usage.py                  # Ejemplo
├── requirements.txt                  # Dependencias
├── README.md                         # Documentación completa
└── QUICKSTART.md                     # Este archivo
```

## Valores de Ejemplo

### Para Arroz
```
N: 90, P: 42, K: 43
Temperatura: 20.88°C, Humedad: 82%, pH: 6.5, Lluvia: 202.94mm
→ Cultivo: RICE
```

### Para Maíz
```
N: 63, P: 35, K: 32
Temperatura: 26.86°C, Humedad: 61.33%, pH: 7.07, Lluvia: 91.91mm
→ Cultivo: MAIZE
```

### Para Trigo
```
N: 69, P: 41, K: 40
Temperatura: 22.75°C, Humedad: 71.2%, pH: 7.19, Lluvia: 74.15mm
→ Cultivo: WHEAT
```

## Rangos de Entrada

| Parámetro | Mín | Máx | Unidad |
|-----------|-----|-----|--------|
| N | 0 | 150 | ppm |
| P | 0 | 150 | ppm |
| K | 0 | 200 | ppm |
| Temperatura | 0 | 50 | °C |
| Humedad | 0 | 100 | % |
| pH | 0 | 14 | - |
| Precipitación | 0 | 500 | mm |

## Solución de Problemas

### El modelo no se encuentra
```
✗ FileNotFoundError: El modelo no se encuentra
Solución: Ejecuta primero python src/train_model.py
```

### Error de importación
```
✗ ModuleNotFoundError
Solución: Instala las dependencias con pip install -r requirements.txt
```

### Streamlit no abre
```bash
# Intenta especificar el puerto
streamlit run app.py --server.port=8501
```

## Próximos Pasos

1. 📖 Lee [README.md](README.md) para documentación completa
2. 🔍 Explora los archivos en `src/` para entender el código
3. 🧪 Experimenta con diferentes valores de entrada
4. 📊 Analiza las visualizaciones generadas

## Rendimiento del Modelo

**Random Forest Classifier**
- Accuracy: ~99.5%
- F1-Score: ~98.5%
- Modelos comparados: Decision Tree, KNN, Naive Bayes

---

¡Listo para empezar! 🎉
