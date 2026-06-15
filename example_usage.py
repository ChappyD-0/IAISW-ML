"""
Script de ejemplo que demuestra cómo usar el sistema de recomendación de cultivos.

Este script muestra:
1. Cómo entrenar el modelo
2. Cómo realizar predicciones
3. Cómo acceder a los análisis
"""

import sys
from pathlib import Path

# Agregar src al path
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path / 'src'))

from train_model import CropRecommendationTrainer
from predict import CropPredictor
from data_analysis import DataAnalyzer
import config


def main():
    """Función principal que ejecuta todos los ejemplos."""
    
    print("=" * 70)
    print("🌾 SISTEMA DE RECOMENDACIÓN DE CULTIVOS - EJEMPLO DE USO")
    print("=" * 70)
    
    # ===== PARTE 1: ANÁLISIS DE DATOS =====
    print("\n" + "-" * 70)
    print("PARTE 1: ANÁLISIS EXPLORATORIO DE DATOS")
    print("-" * 70)
    
    try:
        print("\n¿Desea realizar análisis de datos? (Genera gráficos)")
        print("Nota: Esto genera 4 archivos PNG con visualizaciones")
        user_input = input("\nEjecutar análisis (s/n): ").lower().strip()
        
        if user_input == 's':
            analyzer = DataAnalyzer(str(config.DATA_PATH))
            analyzer.analyze()
        else:
            print("⏭️  Análisis omitido")
    
    except Exception as e:
        print(f"✗ Error en análisis: {e}")
    
    # ===== PARTE 2: ENTRENAMIENTO DEL MODELO =====
    print("\n" + "-" * 70)
    print("PARTE 2: ENTRENAMIENTO DEL MODELO")
    print("-" * 70)
    
    try:
        trainer = CropRecommendationTrainer(
            str(config.DATA_PATH),
            str(config.MODELS_PATH)
        )
        trainer.train()
    
    except Exception as e:
        print(f"✗ Error en entrenamiento: {e}")
        print("\nNo se puede continuar sin modelo entrenado.")
        return
    
    # ===== PARTE 3: PREDICCIONES =====
    print("\n" + "-" * 70)
    print("PARTE 3: REALIZANDO PREDICCIONES")
    print("-" * 70)
    
    try:
        model_path = config.MODELS_PATH / 'crop_model.joblib'
        predictor = CropPredictor(str(model_path))
        
        # Ejemplos de predicción
        examples = [
            {
                'name': 'Ejemplo 1: Arroz (Rice)',
                'params': (90, 42, 43, 20.88, 82.00, 6.50, 202.94)
            },
            {
                'name': 'Ejemplo 2: Maíz (Maize)',
                'params': (63, 35, 32, 26.86, 61.33, 7.07, 91.91)
            },
            {
                'name': 'Ejemplo 3: Trigo (Wheat)',
                'params': (69, 41, 40, 22.75, 71.20, 7.19, 74.15)
            }
        ]
        
        for example in examples:
            print(f"\n{example['name']}")
            print("-" * 50)
            
            N, P, K, temperature, humidity, ph, rainfall = example['params']
            
            try:
                crop = predictor.predict_crop(N, P, K, temperature, humidity, ph, rainfall)
                
                print(f"Parámetros:")
                print(f"  • Nitrógeno (N):    {N} ppm")
                print(f"  • Fósforo (P):      {P} ppm")
                print(f"  • Potasio (K):      {K} ppm")
                print(f"  • Temperatura:      {temperature}°C")
                print(f"  • Humedad:          {humidity}%")
                print(f"  • pH:               {ph}")
                print(f"  • Precipitación:    {rainfall} mm")
                print(f"\n✅ Cultivo recomendado: {crop.upper()}")
            
            except Exception as e:
                print(f"✗ Error en predicción: {e}")
    
    except Exception as e:
        print(f"✗ Error al realizar predicciones: {e}")
    
    # ===== PARTE 4: MODO INTERACTIVO =====
    print("\n" + "-" * 70)
    print("PARTE 4: MODO INTERACTIVO (OPCIONAL)")
    print("-" * 70)
    
    try:
        user_choice = input("\n¿Desea hacer predicciones personalizadas? (s/n): ").lower().strip()
        
        if user_choice == 's':
            model_path = config.MODELS_PATH / 'crop_model.joblib'
            predictor = CropPredictor(str(model_path))
            
            print("\nIngrese los parámetros del suelo y ambientales:")
            print("(Escriba 'salir' para terminar)\n")
            
            while True:
                try:
                    n_input = input("Nitrógeno (N) [0-150]: ").strip()
                    if n_input.lower() == 'salir':
                        break
                    
                    p_input = input("Fósforo (P) [0-150]: ").strip()
                    if p_input.lower() == 'salir':
                        break
                    
                    k_input = input("Potasio (K) [0-200]: ").strip()
                    if k_input.lower() == 'salir':
                        break
                    
                    temp_input = input("Temperatura [0-50]°C: ").strip()
                    if temp_input.lower() == 'salir':
                        break
                    
                    humidity_input = input("Humedad [0-100]%: ").strip()
                    if humidity_input.lower() == 'salir':
                        break
                    
                    ph_input = input("pH [0-14]: ").strip()
                    if ph_input.lower() == 'salir':
                        break
                    
                    rainfall_input = input("Precipitación [0-500]mm: ").strip()
                    if rainfall_input.lower() == 'salir':
                        break
                    
                    # Realizar predicción
                    crop = predictor.predict_crop(
                        float(n_input),
                        float(p_input),
                        float(k_input),
                        float(temp_input),
                        float(humidity_input),
                        float(ph_input),
                        float(rainfall_input)
                    )
                    
                    print(f"\n✅ Cultivo recomendado: {crop.upper()}\n")
                
                except ValueError:
                    print("❌ Por favor, ingrese valores numéricos válidos\n")
                except Exception as e:
                    print(f"❌ Error: {e}\n")
    
    except Exception as e:
        print(f"✗ Error en modo interactivo: {e}")
    
    # ===== FINALIZACIÓN =====
    print("\n" + "=" * 70)
    print("✅ EJEMPLO COMPLETADO")
    print("=" * 70)
    print("""
Para usar el sistema:

1. ENTRENAMIENTO:
   $ python src/train_model.py

2. ANÁLISIS:
   $ python src/data_analysis.py

3. PREDICCIÓN (CLI):
   $ python src/predict.py

4. APLICACIÓN WEB:
   $ streamlit run app.py

Más información en README.md
    """)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Programa interrumpido por el usuario.")
    except Exception as e:
        print(f"\n✗ Error no controlado: {e}")
