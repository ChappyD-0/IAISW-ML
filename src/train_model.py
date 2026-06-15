"""
Módulo para entrenar modelos de recomendación de cultivos.

Este módulo carga el dataset, realiza análisis de datos,
entrena múltiples modelos y selecciona el mejor.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


class CropRecommendationTrainer:
    """Clase para entrenar modelos de recomendación de cultivos."""

    def __init__(self, data_path, models_path):
        """
        Inicializar el entrenador.

        Args:
            data_path (str): Ruta al archivo CSV de datos.
            models_path (str): Ruta donde se guardarán los modelos.
        """
        self.data_path = data_path
        self.models_path = models_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def load_data(self):
        """
        Cargar y validar el dataset.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si faltan columnas requeridas.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"El archivo {self.data_path} no existe.")

        print(f"Cargando datos desde {self.data_path}...")
        self.df = pd.read_csv(self.data_path)

        # Verificar columnas requeridas
        required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        missing_columns = set(required_columns) - set(self.df.columns)

        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas: {missing_columns}")

        print(f"✓ Dataset cargado correctamente: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")

    def analyze_data(self):
        """Analizar y limpiar el dataset."""
        print("\n" + "=" * 50)
        print("ANÁLISIS DEL DATASET")
        print("=" * 50)

        # Información general
        print(f"\nForma del dataset: {self.df.shape}")
        print(f"\nTipos de datos:\n{self.df.dtypes}")

        # Valores nulos
        null_count = self.df.isnull().sum()
        if null_count.sum() == 0:
            print("\n✓ No hay valores nulos en el dataset.")
        else:
            print(f"\nValores nulos por columna:\n{null_count}")

        # Valores duplicados
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count == 0:
            print("✓ No hay filas duplicadas.")
        else:
            print(f"⚠ Se encontraron {duplicate_count} filas duplicadas.")
            self.df = self.df.drop_duplicates()
            print(f"✓ Filas duplicadas removidas. Nuevo tamaño: {self.df.shape[0]} filas")

        # Distribución de cultivos
        print(f"\nDistribución de cultivos (label):")
        print(self.df['label'].value_counts())

        # Estadísticas descriptivas
        print(f"\nEstadísticas descriptivas de variables numéricas:")
        print(self.df.iloc[:, :-1].describe())

    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Separar datos en entrenamiento y prueba.

        Args:
            test_size (float): Proporción de datos para prueba.
            random_state (int): Semilla para reproducibilidad.
        """
        print("\n" + "=" * 50)
        print("PREPARACIÓN DE DATOS")
        print("=" * 50)

        # Separar características y variable objetivo
        X = self.df.drop('label', axis=1)
        y = self.df['label']

        print(f"\nVariables de entrada (X): {X.shape[1]} características")
        print(f"Variable objetivo (y): {y.nunique()} clases")

        # Dividir datos con stratify para mantener el balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"\n✓ Datos divididos correctamente:")
        print(f"  - Entrenamiento: {self.X_train.shape[0]} muestras ({(1-test_size)*100:.0f}%)")
        print(f"  - Prueba: {self.X_test.shape[0]} muestras ({test_size*100:.0f}%)")

    def train_models(self):
        """Entrenar múltiples modelos."""
        print("\n" + "=" * 50)
        print("ENTRENAMIENTO DE MODELOS")
        print("=" * 50)

        # Decision Tree
        print("\n[1/4] Entrenando Decision Tree...")
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(self.X_train, self.y_train)
        dt_pred = dt_model.predict(self.X_test)
        dt_accuracy = accuracy_score(self.y_test, dt_pred)
        self.models['Decision Tree'] = dt_model
        self.results['Decision Tree'] = dt_accuracy
        print(f"      ✓ Accuracy: {dt_accuracy:.4f}")

        # KNN
        print("[2/4] Entrenando K-Nearest Neighbors...")
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(self.X_train, self.y_train)
        knn_pred = knn_model.predict(self.X_test)
        knn_accuracy = accuracy_score(self.y_test, knn_pred)
        self.models['KNN'] = knn_model
        self.results['KNN'] = knn_accuracy
        print(f"      ✓ Accuracy: {knn_accuracy:.4f}")

        # Naive Bayes
        print("[3/4] Entrenando Naive Bayes...")
        nb_model = GaussianNB()
        nb_model.fit(self.X_train, self.y_train)
        nb_pred = nb_model.predict(self.X_test)
        nb_accuracy = accuracy_score(self.y_test, nb_pred)
        self.models['Naive Bayes'] = nb_model
        self.results['Naive Bayes'] = nb_accuracy
        print(f"      ✓ Accuracy: {nb_accuracy:.4f}")

        # Random Forest
        print("[4/4] Entrenando Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        rf_pred = rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = rf_accuracy
        print(f"      ✓ Accuracy: {rf_accuracy:.4f}")

    def compare_models(self):
        """Comparar el rendimiento de los modelos."""
        print("\n" + "=" * 50)
        print("COMPARACIÓN DE MODELOS")
        print("=" * 50)

        # Ordenar modelos por accuracy
        sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)

        print("\nRanking de modelos por Accuracy:\n")
        for i, (model_name, accuracy) in enumerate(sorted_results, 1):
            print(f"{i}. {model_name:20s} → Accuracy: {accuracy:.4f}")

        # Seleccionar mejor modelo
        self.best_model_name = sorted_results[0][0]
        self.best_model = self.models[self.best_model_name]

        print(f"\n✓ Mejor modelo: {self.best_model_name}")

    def evaluate_best_model(self):
        """Evaluar el mejor modelo en detalle."""
        print("\n" + "=" * 50)
        print(f"EVALUACIÓN DETALLADA: {self.best_model_name.upper()}")
        print("=" * 50)

        # Predicciones
        y_pred = self.best_model.predict(self.X_test)

        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")

        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nMatriz de Confusión:")
        print(cm)

        # Reporte de clasificación
        print(f"\nReporte de Clasificación:")
        print(classification_report(self.y_test, y_pred))

        # Visualizar matriz de confusión
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.best_model.classes_)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Matriz de Confusión - {self.best_model_name}')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Matriz de confusión guardada en 'confusion_matrix.png'")

    def save_best_model(self):
        """Guardar el mejor modelo en formato joblib."""
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        model_filename = os.path.join(self.models_path, 'crop_model.joblib')
        joblib.dump(self.best_model, model_filename)
        print(f"\n✓ Mejor modelo guardado en: {model_filename}")

    def train(self):
        """Ejecutar el flujo completo de entrenamiento."""
        try:
            self.load_data()
            self.analyze_data()
            self.prepare_data()
            self.train_models()
            self.compare_models()
            self.evaluate_best_model()
            self.save_best_model()

            print("\n" + "=" * 50)
            print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            print("=" * 50)

        except Exception as e:
            print(f"\n✗ Error durante el entrenamiento: {str(e)}")
            raise


def main():
    """Función principal."""
    # Rutas
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data' / 'Crop_recommendation.csv'
    models_path = base_path / 'models'

    # Crear entrenador y ejecutar
    trainer = CropRecommendationTrainer(str(data_path), str(models_path))
    trainer.train()


if __name__ == '__main__':
    main()
