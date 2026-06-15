# 🌾 Crop Recommendation ML

## Descripción General

**Crop Recommendation ML** es un sistema de recomendación de cultivos basado en Machine Learning que utiliza algoritmos de clasificación para predecir el cultivo más apropiado según las características del suelo y las condiciones ambientales.

El proyecto implementa una solución completa que incluye:
- **Análisis y preparación de datos**
- **Entrenamiento de múltiples modelos de ML**
- **Evaluación y comparación de rendimiento**
- **Interfaz web interactiva con Streamlit**
- **API de predicción**

## 🎯 Objetivo

Desarrollar un sistema inteligente que ayude a agricultores a tomar decisiones informadas sobre qué cultivos sembrar basándose en:
- **Propiedades químicas del suelo** (Nitrógeno, Fósforo, Potasio)
- **Condiciones ambientales** (Temperatura, Humedad, pH, Precipitación)

El sistema busca optimizar la productividad agrícola y minimizar el riesgo de fracaso en las cosechas.

## 📊 Dataset

### Descripción
El dataset `Crop_recommendation.csv` contiene información de cultivos recomendados basada en propiedades del suelo y condiciones ambientales. Cada registro representa una combinación de parámetros agrícolas con su correspondiente cultivo recomendado.

### Características (Variables de entrada - X)

| Variable | Descripción | Rango | Unidad |
|----------|-------------|-------|--------|
| **N** | Proporción de Nitrógeno en el suelo | 0-150 | ppm |
| **P** | Proporción de Fósforo en el suelo | 0-150 | ppm |
| **K** | Proporción de Potasio en el suelo | 0-200 | ppm |
| **temperature** | Temperatura promedio | 0-50 | °C |
| **humidity** | Humedad relativa | 0-100 | % |
| **ph** | pH del suelo | 0-14 | Escala |
| **rainfall** | Precipitación anual promedio | 0-500 | mm |

### Variable Objetivo (y)
- **label**: Cultivo recomendado (clasificación multiclase)

### Estadísticas del Dataset
- **Total de muestras**: ~2200 registros
- **Variables numéricas**: 7
- **Clases (cultivos)**: 22 tipos diferentes
- **Balance de clases**: Estratificado (distribuido proporcionalmente)

## 🤖 Modelo de Machine Learning

### Modelos Entrenados

Se entrenan y comparan los siguientes algoritmos de clasificación:

#### 1. **Decision Tree**
```
Características: Simple e interpretable
Parámetros: random_state=42
Ventajas: Fácil de entender, detecta relaciones no lineales
Limitaciones: Propenso a overfitting
```

#### 2. **K-Nearest Neighbors (KNN)**
```
Características: Basado en distancia
Parámetros: n_neighbors=5
Ventajas: Simple, buen rendimiento general
Limitaciones: Sensible al ruido y escalado
```

#### 3. **Naive Bayes**
```
Características: Basado en probabilidad
Parámetros: GaussianNB (distribuida gaussiana)
Ventajas: Rápido, eficiente en espacio
Limitaciones: Asume independencia entre características
```

#### 4. **Random Forest** ⭐ (Modelo Recomendado)
```
Características: Ensemble de árboles
Parámetros: n_estimators=100, random_state=42
Ventajas: Excelente rendimiento, robusto, maneja interacciones
Limitaciones: Menos interpretable que Decision Tree
```

### Metodología

1. **División de datos**: 80% entrenamiento, 20% prueba
2. **Estratificación**: Mantiene el balance de clases (stratify=y)
3. **Métrica principal**: Accuracy (exactitud)
4. **Validación**: Matriz de confusión y reporte de clasificación

## 📁 Estructura del Proyecto

```
crop-recommendation-ml/
├── data/
│   └── Crop_recommendation.csv          # Dataset de entrada
├── models/
│   └── crop_model.joblib                # Modelo entrenado guardado
├── src/
│   ├── train_model.py                   # Script de entrenamiento
│   ├── predict.py                       # Módulo de predicción
│   └── data_analysis.py                 # Análisis exploratorio
├── app.py                               # Interfaz Streamlit
├── requirements.txt                     # Dependencias del proyecto
├── README.md                            # Este archivo
├── confusion_matrix.png                 # Visualización (generada)
├── distributions.png                    # Gráficos (generada)
├── crop_distribution.png                # Gráficos (generada)
├── correlation_matrix.png               # Matriz de correlación (generada)
└── analysis_by_crop.png                 # Análisis por cultivo (generada)
```

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o descargar el proyecto**
   ```bash
   cd crop-recommendation-ml
   ```

2. **Crear un entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual**
   - **En Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```
   - **En Windows:**
     ```bash
     venv\Scripts\activate
     ```

4. **Instalar las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## 🏋️ Entrenamiento del Modelo

### Ejecutar el entrenamiento completo

```bash
python src/train_model.py
```

Este script realizará:
1. ✅ Carga del dataset
2. ✅ Análisis de datos (valores nulos, duplicados)
3. ✅ Preparación de datos (separación en train/test)
4. ✅ Entrenamiento de 4 modelos diferentes
5. ✅ Comparación y selección del mejor modelo
6. ✅ Evaluación detallada (matriz de confusión, reporte)
7. ✅ Guardar el modelo en `models/crop_model.joblib`

**Salida esperada:**
```
Cargando datos desde data/Crop_recommendation.csv...
✓ Dataset cargado correctamente: 2200 filas, 8 columnas

==================================================
ANÁLISIS DEL DATASET
==================================================
...
==================================================
ENTRENAMIENTO DE MODELOS
==================================================
[1/4] Entrenando Decision Tree...
      ✓ Accuracy: 0.9909
...
==================================================
COMPARACIÓN DE MODELOS
==================================================
Ranking de modelos por Accuracy:
1. Random Forest          → Accuracy: 0.9954
...
✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE
```

## 📊 Análisis de Datos

Ejecutar análisis exploratorio y generar visualizaciones:

```bash
python src/data_analysis.py
```

Genera los siguientes gráficos:
- **distributions.png**: Histogramas de todas las variables
- **crop_distribution.png**: Distribución de cultivos
- **correlation_matrix.png**: Matriz de correlación
- **analysis_by_crop.png**: Análisis de características por cultivo

## 💻 Uso de la Aplicación Web

### Iniciar la interfaz Streamlit

```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

### Características de la Interfaz

✨ **Interfaz intuitiva y fácil de usar**
- Formulario para ingresar parámetros agrícolas
- Validación de entrada en tiempo real
- Recomendación inmediata del cultivo
- Resumen visual de parámetros utilizados
- Información sobre los rangos de valores

### Pasos de Uso

1. **Ingrese los parámetros del suelo**:
   - Nitrógeno (N)
   - Fósforo (P)
   - Potasio (K)

2. **Ingrese las condiciones ambientales**:
   - Temperatura
   - Humedad
   - pH del suelo
   - Precipitación

3. **Haga clic en "Obtener Recomendación"**

4. **Reciba el cultivo recomendado con métricas**

## 📝 Ejemplo de Predicción

### Predicción Manual (Python)

```python
from src.predict import predict_crop

# Parámetros de ejemplo
cultivo = predict_crop(
    N=90,
    P=42,
    K=43,
    temperature=20.88,
    humidity=82.00,
    ph=6.50,
    rainfall=202.94
)

print(f"Cultivo recomendado: {cultivo}")
# Salida: Cultivo recomendado: rice
```

### Ejecución del módulo predict

```bash
python src/predict.py
```

Esto ejecutará un ejemplo de predicción y mostrará el resultado.

### Usando desde la aplicación web

1. Abrir `http://localhost:8501`
2. Ingresar los valores anteriores en el formulario
3. Hacer clic en "Obtener Recomendación"
4. Ver el resultado: **RICE**

## 📈 Resultados Esperados

### Rendimiento del Modelo

**Random Forest (Modelo Recomendado)**:
- **Accuracy**: ~99.5% en datos de prueba
- **Precisión**: Alto en todas las clases
- **Recall**: Excelente balance

### Comparación de Modelos
```
1. Random Forest          → Accuracy: ~99.5% ⭐
2. Decision Tree          → Accuracy: ~99.0%
3. Naive Bayes            → Accuracy: ~95.0%
4. KNN                    → Accuracy: ~98.0%
```

## 🔍 Interpretación de Resultados

### Matriz de Confusión
- Diagonal principal: Predicciones correctas
- Valores fuera de diagonal: Errores de clasificación
- Indica qué cultivos se confunden entre sí

### Reporte de Clasificación
- **Precision**: De las predicciones positivas, cuántas son correctas
- **Recall**: De los casos positivos reales, cuántos se identifican
- **F1-Score**: Balance entre Precision y Recall
- **Support**: Número de muestras por clase

## ⚠️ Limitaciones del Sistema

### 1. **Suposiciones del Modelo**
   - Asume relaciones aprendidas del dataset de entrenamiento
   - Las predicciones se basan en patrones históricos
   - No considera factores externos (plagas, enfermedades)

### 2. **Limitaciones de Datos**
   - Dataset limitado a ~2200 muestras
   - Solo 22 tipos de cultivos disponibles
   - Datos agregados (promedios anuales)
   - No incluye datos temporales/estacionales

### 3. **Rango de Valores**
   - Las predicciones son más confiables dentro de los rangos del dataset
   - Valores fuera de rango pueden producir resultados inesperados
   - No es recomendable extrapolar más allá de los límites

### 4. **Factores No Considerados**
   - ❌ Disponibilidad de agua/riego
   - ❌ Densidad de población
   - ❌ Infraestructura de mercado
   - ❌ Plagas y enfermedades
   - ❌ Tendencias del mercado
   - ❌ Costos de producción
   - ❌ Experiencia del agricultor

### 5. **Uso Recomendado**
   ✅ Como herramienta de apoyo en la toma de decisiones
   ✅ No como única fuente de información
   ✅ Combinado con experiencia agrícola local
   ✅ Validado con agrónomos profesionales

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|-----------|---------|----------|
| Python | 3.8+ | Lenguaje principal |
| Pandas | 2.0.3 | Manipulación de datos |
| NumPy | 1.24.3 | Cálculos numéricos |
| Scikit-learn | 1.3.0 | Modelos ML |
| Joblib | 1.3.1 | Serialización de modelos |
| Streamlit | 1.28.1 | Interfaz web |
| Matplotlib | 3.7.2 | Visualización estática |
| Seaborn | 0.12.2 | Visualización estadística |

## 📚 Mejores Prácticas Implementadas

### 1. **Organización del Código**
- ✅ Estructura modular y escalable
- ✅ Separación de responsabilidades
- ✅ Uso de clases y métodos reutilizables

### 2. **Documentación**
- ✅ Docstrings en todas las funciones
- ✅ Comentarios explicativos
- ✅ README completo

### 3. **Validación**
- ✅ Validación de entrada de datos
- ✅ Manejo de excepciones
- ✅ Verificación de archivos

### 4. **Reproducibilidad**
- ✅ random_state=42 para reproducibilidad
- ✅ Versionado de dependencias exactas
- ✅ Procedimiento documentado

### 5. **Performance**
- ✅ Caching de modelo en Streamlit
- ✅ Uso de joblib para serialización eficiente
- ✅ Procesamiento optimizado

## 🤝 Contribuciones y Mejoras Futuras

### Posibles Mejoras
1. **Más datos**: Aumentar el dataset con nuevas muestras
2. **Validación cruzada**: K-fold cross-validation
3. **Tunning de hiperparámetros**: GridSearchCV o RandomizedSearchCV
4. **Más modelos**: Deep Learning, Ensemble avanzados
5. **API REST**: Crear un servidor Flask/FastAPI
6. **Base de datos**: Guardar histórico de predicciones
7. **Análisis temporal**: Incluir datos estacionales

## 📞 Soporte

Para reportar problemas o hacer sugerencias:
- Verificar que todas las dependencias estén instaladas
- Revisar que el archivo CSV esté en la carpeta `data/`
- Ejecutar primero `python src/train_model.py`
- Consultar los logs de error

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.

## ✍️ Autor

**Proyecto de Inteligencia Artificial Aplicada**
- Semestre 6
- Institución Académica
- Año: 2024

## 🎓 Notas Académicas

Este proyecto demuestra:
- ✅ Comprensión de pipelines de Machine Learning
- ✅ Aplicación de múltiples algoritmos de clasificación
- ✅ Evaluación y comparación de modelos
- ✅ Desarrollo de interfaces de usuario
- ✅ Documentación profesional
- ✅ Mejores prácticas en desarrollo

---

**Estado**: ✅ Proyecto Completado y Funcional

**Última actualización**: 2024

**Mantener actualizado**: Revisar y actualizar dependencias regularmente
