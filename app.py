"""
Aplicación web para recomendación de cultivos usando Streamlit.

Esta aplicación permite a los usuarios ingresar parámetros agrícolas
y recibir recomendaciones de cultivos basadas en un modelo ML.
"""

import streamlit as st
from pathlib import Path
import sys
import os

# Agregar src al path para importar módulos
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path / 'src'))

from predict import CropPredictor

# Configuración de la página
st.set_page_config(
    page_title="Recomendador de Cultivos",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Cargar el modelo una sola vez."""
    model_path = base_path / 'models' / 'crop_model.joblib'
    
    if not model_path.exists():
        return None
    
    try:
        predictor = CropPredictor(str(model_path))
        return predictor
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None


def validate_inputs(N, P, K, temperature, humidity, ph, rainfall):
    """Validar que los inputs sean válidos."""
    errors = []
    
    # Validar que no estén vacíos
    if N == "" or P == "" or K == "":
        errors.append("Los campos N, P, K no pueden estar vacíos")
    
    if temperature == "" or humidity == "" or ph == "" or rainfall == "":
        errors.append("Los campos de condiciones ambientales no pueden estar vacíos")
    
    # Validar que sean numéricos
    try:
        if N != "":
            n_val = float(N)
            if not (0 <= n_val <= 150):
                errors.append("Nitrógeno (N) debe estar entre 0 y 150")
        
        if P != "":
            p_val = float(P)
            if not (0 <= p_val <= 150):
                errors.append("Fósforo (P) debe estar entre 0 y 150")
        
        if K != "":
            k_val = float(K)
            if not (0 <= k_val <= 200):
                errors.append("Potasio (K) debe estar entre 0 y 200")
        
        if temperature != "":
            temp_val = float(temperature)
            if not (0 <= temp_val <= 50):
                errors.append("Temperatura debe estar entre 0 y 50 °C")
        
        if humidity != "":
            hum_val = float(humidity)
            if not (0 <= hum_val <= 100):
                errors.append("Humedad debe estar entre 0 y 100 %")
        
        if ph != "":
            ph_val = float(ph)
            if not (0 <= ph_val <= 14):
                errors.append("pH debe estar entre 0 y 14")
        
        if rainfall != "":
            rain_val = float(rainfall)
            if not (0 <= rain_val <= 500):
                errors.append("Precipitación debe estar entre 0 y 500 mm")
    
    except ValueError:
        errors.append("Por favor, ingrese valores numéricos válidos")
    
    return errors


def main():
    """Función principal de la aplicación."""
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>🌾 Recomendador de Cultivos</h1>
        <p>Sistema de recomendación de cultivos usando Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar modelo
    predictor = load_model()
    
    if predictor is None:
        st.error("""
        ❌ No se encontró el modelo entrenado.
        
        Por favor, ejecute primero:
        ```
        python src/train_model.py
        ```
        """)
        return
    
    # Contenido principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Parámetros del Suelo")
        
        # Inputs de características del suelo
        with st.form("soil_form"):
            st.markdown("**Nutrientes (NPK)**")
            n_input = st.text_input("Nitrógeno (N)", value="", help="Rango: 0-150")
            p_input = st.text_input("Fósforo (P)", value="", help="Rango: 0-150")
            k_input = st.text_input("Potasio (K)", value="", help="Rango: 0-200")
            
            st.markdown("**Condiciones Ambientales**")
            temp_input = st.text_input("Temperatura (°C)", value="", help="Rango: 0-50")
            humidity_input = st.text_input("Humedad (%)", value="", help="Rango: 0-100")
            ph_input = st.text_input("pH del suelo", value="", help="Rango: 0-14")
            rainfall_input = st.text_input("Precipitación (mm)", value="", help="Rango: 0-500")
            
            submitted = st.form_submit_button("🔍 Obtener Recomendación", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Información de Uso")
        
        st.info("""
        **Cómo usar la aplicación:**
        
        1. Ingrese los valores de los nutrientes del suelo (N, P, K)
        2. Ingrese las condiciones ambientales (temperatura, humedad, pH, precipitación)
        3. Haga clic en "Obtener Recomendación"
        4. El sistema le mostrará el cultivo recomendado
        
        **Ranges de entrada:**
        - **N (Nitrógeno):** 0-150
        - **P (Fósforo):** 0-150
        - **K (Potasio):** 0-200
        - **Temperatura:** 0-50 °C
        - **Humedad:** 0-100 %
        - **pH:** 0-14
        - **Precipitación:** 0-500 mm
        """)
        
        if submitted:
            # Validar inputs
            errors = validate_inputs(n_input, p_input, k_input, temp_input, 
                                    humidity_input, ph_input, rainfall_input)
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                try:
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
                    
                    st.success(f"✅ **Cultivo Recomendado: {crop.upper()}**")
                    
                    # Mostrar resumen
                    st.markdown("### 📈 Resumen de Parámetros")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("Nitrógeno (N)", f"{float(n_input):.1f}")
                    with col_b:
                        st.metric("Fósforo (P)", f"{float(p_input):.1f}")
                    with col_c:
                        st.metric("Potasio (K)", f"{float(k_input):.1f}")
                    with col_d:
                        st.metric("Temperatura", f"{float(temp_input):.1f}°C")
                    
                    col_e, col_f, col_g, col_h = st.columns(4)
                    
                    with col_e:
                        st.metric("Humedad", f"{float(humidity_input):.1f}%")
                    with col_f:
                        st.metric("pH", f"{float(ph_input):.2f}")
                    with col_g:
                        st.metric("Precipitación", f"{float(rainfall_input):.1f}mm")
                    with col_h:
                        st.metric("Resultado", crop.upper(), delta=None)
                
                except Exception as e:
                    st.error(f"❌ Error en la predicción: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Sistema de Recomendación de Cultivos | Powered by Machine Learning</p>
        <p><small>© 2024 - Inteligencia Artificial Aplicada</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
