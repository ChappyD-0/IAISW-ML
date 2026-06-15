#!/bin/bash

# Script de inicialización rápida para el proyecto Crop Recommendation ML

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  🌾 CROP RECOMMENDATION ML - INICIALIZADOR DEL PROYECTO          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Detectar el sistema operativo
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    ACTIVATE="source venv/bin/activate"
    PYTHON="python"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    ACTIVATE="venv\Scripts\activate"
    PYTHON="python"
else
    echo "Sistema operativo no reconocido"
    exit 1
fi

# Mostrar opciones
echo "¿Qué deseas hacer?"
echo ""
echo "1) Ejecutar la aplicación web (Streamlit) - RECOMENDADO"
echo "2) Entrenar el modelo"
echo "3) Analizar los datos"
echo "4) Realizar predicción (CLI)"
echo "5) Ejecutar script de ejemplo"
echo "6) Activar entorno virtual"
echo ""

read -p "Selecciona una opción (1-6): " option

case $option in
    1)
        echo "🚀 Iniciando aplicación web..."
        echo ""
        eval $ACTIVATE
        streamlit run app.py
        ;;
    2)
        echo "🏋️ Entrenando modelo..."
        echo ""
        eval $ACTIVATE
        $PYTHON src/train_model.py
        ;;
    3)
        echo "📊 Analizando datos..."
        echo ""
        eval $ACTIVATE
        $PYTHON src/data_analysis.py
        ;;
    4)
        echo "🔍 Realizando predicción..."
        echo ""
        eval $ACTIVATE
        $PYTHON src/predict.py
        ;;
    5)
        echo "📝 Ejecutando ejemplo..."
        echo ""
        eval $ACTIVATE
        $PYTHON example_usage.py
        ;;
    6)
        echo "✅ Activando entorno virtual..."
        echo "Ejecuta: $ACTIVATE"
        eval $ACTIVATE
        ;;
    *)
        echo "❌ Opción inválida"
        exit 1
        ;;
esac
