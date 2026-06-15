"""
Módulo para realizar predicciones con el modelo entrenado.

Este módulo carga el modelo guardado y realiza predicciones
de cultivos recomendados basado en parámetros agrícolas.
"""

import joblib
import os
from pathlib import Path
import numpy as np


class CropPredictor:
    """Clase para realizar predicciones de cultivos."""

    def __init__(self, model_path):
        """
        Inicializar el predictor.

        Args:
            model_path (str): Ruta al modelo entrenado (joblib).

        Raises:
            FileNotFoundError: Si el modelo no existe.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo no se encuentra en {model_path}")

        self.model_path = model_path
        self.model = joblib.load(model_path)
        print(f"✓ Modelo cargado correctamente desde: {model_path}")

    def predict_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Predecir el cultivo recomendado.

        Args:
            N (float): Contenido de Nitrógeno en el suelo.
            P (float): Contenido de Fósforo en el suelo.
            K (float): Contenido de Potasio en el suelo.
            temperature (float): Temperatura en grados Celsius.
            humidity (float): Humedad en porcentaje.
            ph (float): pH del suelo.
            rainfall (float): Precipitación en mm.

        Returns:
            str: Cultivo recomendado.

        Raises:
            ValueError: Si alguno de los parámetros es inválido.
        """
        try:
            # Convertir a float y validar
            features = [N, P, K, temperature, humidity, ph, rainfall]
            features = [float(f) for f in features]

            # Validar rangos razonables
            self._validate_inputs(features)

            # Crear array para predicción
            features_array = np.array([features])

            # Realizar predicción
            prediction = self.model.predict(features_array)[0]

            return prediction

        except ValueError as e:
            raise ValueError(f"Error en la predicción: {str(e)}")

    @staticmethod
    def _validate_inputs(features):
        """
        Validar que los inputs estén dentro de rangos razonables.

        Args:
            features (list): Lista de características.

        Raises:
            ValueError: Si algún valor está fuera de rango.
        """
        N, P, K, temperature, humidity, ph, rainfall = features

        # Validaciones básicas
        if not (0 <= N <= 150):
            raise ValueError("Nitrógeno (N) debe estar entre 0 y 150")
        if not (0 <= P <= 150):
            raise ValueError("Fósforo (P) debe estar entre 0 y 150")
        if not (0 <= K <= 200):
            raise ValueError("Potasio (K) debe estar entre 0 y 200")
        if not (0 <= temperature <= 50):
            raise ValueError("Temperatura debe estar entre 0 y 50 °C")
        if not (0 <= humidity <= 100):
            raise ValueError("Humedad debe estar entre 0 y 100 %")
        if not (0 <= ph <= 14):
            raise ValueError("pH debe estar entre 0 y 14")
        if not (0 <= rainfall <= 500):
            raise ValueError("Precipitación debe estar entre 0 y 500 mm")


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Función standalone para realizar predicciones.

    Esta función carga el modelo y realiza la predicción
    en una sola llamada.

    Args:
        N (float): Contenido de Nitrógeno en el suelo.
        P (float): Contenido de Fósforo en el suelo.
        K (float): Contenido de Potasio en el suelo.
        temperature (float): Temperatura en grados Celsius.
        humidity (float): Humedad en porcentaje.
        ph (float): pH del suelo.
        rainfall (float): Precipitación en mm.

    Returns:
        str: Cultivo recomendado.

    Raises:
        Exception: Si hay algún error en la predicción.
    """
    base_path = Path(__file__).parent.parent
    model_path = base_path / 'models' / 'crop_model.joblib'

    predictor = CropPredictor(str(model_path))
    return predictor.predict_crop(N, P, K, temperature, humidity, ph, rainfall)


def main():
    """Función principal para pruebas."""
    try:
        base_path = Path(__file__).parent.parent
        model_path = base_path / 'models' / 'crop_model.joblib'

        predictor = CropPredictor(str(model_path))

        # Ejemplo de predicción
        print("\nRealizando predicción de ejemplo...")
        print("-" * 50)

        N = 90
        P = 42
        K = 43
        temperature = 20.88
        humidity = 82.00
        ph = 6.50
        rainfall = 202.94

        result = predictor.predict_crop(N, P, K, temperature, humidity, ph, rainfall)

        print(f"Parámetros:")
        print(f"  Nitrógeno (N): {N}")
        print(f"  Fósforo (P): {P}")
        print(f"  Potasio (K): {K}")
        print(f"  Temperatura: {temperature}°C")
        print(f"  Humedad: {humidity}%")
        print(f"  pH: {ph}")
        print(f"  Precipitación: {rainfall}mm")
        print(f"\n✓ Cultivo recomendado: {result}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")


if __name__ == '__main__':
    main()
