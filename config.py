"""
Configuración centralizada del proyecto.
"""

from pathlib import Path

# Rutas del proyecto
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data' / 'Crop_recommendation.csv'
MODELS_PATH = BASE_PATH / 'models'
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Configuración de modelos
RANDOM_STATE = 42
TEST_SIZE = 0.2
STRATIFY = True

# Configuración de modelos específicos
DECISION_TREE_CONFIG = {
    'random_state': RANDOM_STATE
}

KNN_CONFIG = {
    'n_neighbors': 5
}

NAIVE_BAYES_CONFIG = {}

RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE
}

# Columnas del dataset
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COLUMN = 'label'

# Rangos válidos de entrada
INPUT_RANGES = {
    'N': (0, 150),
    'P': (0, 150),
    'K': (0, 200),
    'temperature': (0, 50),
    'humidity': (0, 100),
    'ph': (0, 14),
    'rainfall': (0, 500)
}

# Mensajes
MESSAGES = {
    'model_loaded': '✓ Modelo cargado correctamente',
    'data_loaded': '✓ Datos cargados correctamente',
    'training_complete': '✓ Entrenamiento completado exitosamente',
    'prediction_success': '✓ Predicción realizada correctamente',
    'model_saved': '✓ Modelo guardado correctamente'
}

ERRORS = {
    'model_not_found': 'El modelo no se encontró',
    'data_not_found': 'Los datos no se encontraron',
    'invalid_input': 'Entrada inválida',
    'training_error': 'Error durante el entrenamiento'
}
