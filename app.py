"""
SERVIDOR WEB SIMPLE PARA IA DE DIBUJOS
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64
from PIL import Image
import io
import os

app = Flask(__name__)

# ============================================================================
# CARGAR MODELO
# ============================================================================

print("🎨 Iniciando servidor...")

# Verificar que existe el modelo
if not os.path.exists('modelo_dibujos.keras'):
    print("❌ ERROR: No se encuentra el archivo 'modelo_dibujos.keras'")
    print("   Copia tu modelo entrenado a esta carpeta")
    exit()

# Cargar modelo
modelo = tf.keras.models.load_model('modelo_dibujos.keras')
print("✅ Modelo cargado correctamente")

# Clases (deben coincidir con las que usaste para entrenar)
CLASES = ["circulo", "cuadrado", "triangulo", "casa"]
print(f"🎯 Clases: {', '.join(CLASES)}")

# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================

def predecir_dibujo(imagen_data):
    """
    Procesa la imagen y devuelve la predicción
    """
    try:
        # 1. Decodificar imagen base64
        if 'base64,' in imagen_data:
            imagen_data = imagen_data.split('base64,')[1]
        
        imagen_bytes = base64.b64decode(imagen_data)
        imagen_pil = Image.open(io.BytesIO(imagen_bytes))
        
        # 2. Convertir a numpy array
        imagen_np = np.array(imagen_pil)
        
        # 3. Convertir a escala de grises
        if len(imagen_np.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
        else:
            imagen_gris = imagen_np
        
        # 4. Invertir colores (la IA se entrenó con fondo negro)
        imagen_invertida = cv2.bitwise_not(imagen_gris)
        
        # 5. Redimensionar a 28x28
        imagen_pequena = cv2.resize(imagen_invertida, (28, 28))
        
        # 6. Normalizar (0-1)
        imagen_normalizada = imagen_pequena.astype('float32') / 255.0
        
        # 7. Añadir dimensiones para el modelo
        imagen_final = imagen_normalizada.reshape(1, 28, 28, 1)
        
        # 8. Predecir
        predicciones = modelo.predict(imagen_final, verbose=0)[0]
        clase_idx = int(np.argmax(predicciones))
        confianza = float(predicciones[clase_idx])
        
        # 9. Preparar resultado
        resultado = {
            "exito": True,
            "clase": CLASES[clase_idx],
            "confianza": confianza,
            "porcentaje": f"{confianza*100:.1f}%"
        }
        
        return resultado
        
    except Exception as e:
        return {
            "exito": False,
            "error": str(e)
        }

# ============================================================================
# RUTAS DE LA WEB
# ============================================================================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', clases=CLASES)

@app.route('/predecir', methods=['POST'])
def predict():
    """API para predicciones"""
    data = request.get_json()
    
    if 'imagen' not in data:
        return jsonify({"exito": False, "error": "No se recibió imagen"})
    
    resultado = predecir_dibujo(data['imagen'])
    return jsonify(resultado)

# ============================================================================
# INICIAR SERVIDOR
# ============================================================================

if __name__ == '__main__':
    print("\n🚀 Servidor listo!")
    print("📱 Abre tu navegador en: http://localhost:5000")
    print("🔴 Presiona CTRL+C para detener")
    print("=" * 50)
    app.run(debug=True, port=5000)