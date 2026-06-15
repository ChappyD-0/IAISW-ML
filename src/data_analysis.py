"""
Módulo para análisis exploratorio de datos.

Este módulo realiza análisis visual y estadístico
del dataset de recomendación de cultivos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


class DataAnalyzer:
    """Clase para análisis de datos."""

    def __init__(self, data_path):
        """
        Inicializar el analizador.

        Args:
            data_path (str): Ruta al archivo CSV.
        """
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Cargar el dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"El archivo {self.data_path} no existe.")

        self.df = pd.read_csv(self.data_path)
        print(f"✓ Dataset cargado: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")

    def summary_statistics(self):
        """Mostrar estadísticas descriptivas."""
        print("\n" + "=" * 50)
        print("ESTADÍSTICAS DESCRIPTIVAS")
        print("=" * 50)

        print("\nInformación del dataset:")
        print(f"  Formas: {self.df.shape}")
        print(f"  Tipos de datos:\n{self.df.dtypes}")

        print("\nEstadísticas numéricas:")
        print(self.df.describe())

        print("\nDistribución de cultivos:")
        print(self.df['label'].value_counts())

    def visualize_distributions(self):
        """Crear visualizaciones de distribuciones."""
        print("\nGenerando visualizaciones de distribuciones...")

        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)

        # Crear subgráficos para variables numéricas
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for idx, feature in enumerate(features):
            axes[idx].hist(self.df[feature], bins=30, color='steelblue', edgecolor='black')
            axes[idx].set_title(f'Distribución de {feature}', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frecuencia')

        # Ocultar el último subplot (tenemos 7 features y 8 subplots)
        axes[-1].remove()

        plt.tight_layout()
        plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico de distribuciones guardado en 'distributions.png'")

    def visualize_crop_distribution(self):
        """Visualizar la distribución de cultivos."""
        print("Generando gráfico de distribución de cultivos...")

        plt.figure(figsize=(12, 6))

        crop_counts = self.df['label'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))

        plt.subplot(1, 2, 1)
        crop_counts.plot(kind='bar', color=colors, edgecolor='black')
        plt.title('Cantidad de muestras por cultivo', fontsize=12, fontweight='bold')
        plt.xlabel('Cultivo')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=45, ha='right')

        plt.subplot(1, 2, 2)
        crop_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Proporción de muestras por cultivo', fontsize=12, fontweight='bold')
        plt.ylabel('')

        plt.tight_layout()
        plt.savefig('crop_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico de distribución de cultivos guardado en 'crop_distribution.png'")

    def visualize_correlations(self):
        """Crear matriz de correlación."""
        print("Generando matriz de correlación...")

        plt.figure(figsize=(10, 8))

        # Calcular correlaciones (excluyendo la columna 'label')
        numeric_df = self.df.drop('label', axis=1)
        correlation_matrix = numeric_df.corr()

        # Crear heatmap
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": 0.8})

        plt.title('Matriz de Correlación - Variables Numéricas', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Matriz de correlación guardada en 'correlation_matrix.png'")

    def analyze_by_crop(self):
        """Análisis de características por cultivo."""
        print("Realizando análisis por cultivo...")

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        crops = self.df['label'].unique()

        for idx, feature in enumerate(features):
            for crop in crops:
                data = self.df[self.df['label'] == crop][feature]
                axes[idx].hist(data, bins=20, alpha=0.5, label=crop)

            axes[idx].set_title(f'{feature} por cultivo', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frecuencia')
            axes[idx].legend(fontsize=7, loc='best')

        # Ocultar el último subplot
        axes[-1].remove()

        plt.tight_layout()
        plt.savefig('analysis_by_crop.png', dpi=300, bbox_inches='tight')
        print("✓ Análisis por cultivo guardado en 'analysis_by_crop.png'")

    def analyze(self):
        """Ejecutar análisis completo."""
        try:
            self.load_data()
            self.summary_statistics()
            self.visualize_distributions()
            self.visualize_crop_distribution()
            self.visualize_correlations()
            self.analyze_by_crop()

            print("\n" + "=" * 50)
            print("✓ ANÁLISIS COMPLETADO")
            print("=" * 50)

        except Exception as e:
            print(f"\n✗ Error durante el análisis: {str(e)}")
            raise


def main():
    """Función principal."""
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data' / 'Crop_recommendation.csv'

    analyzer = DataAnalyzer(str(data_path))
    analyzer.analyze()


if __name__ == '__main__':
    main()
