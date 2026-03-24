"""
ENTRENADOR DE IA PARA RECONOCER DIBUJOS
Este programa entrena una red neuronal con tus dibujos
"""

import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("🤖 ENTRENADOR DE IA PARA DIBUJOS")
print("=" * 40)

# Configuración
TAMAÑO_IMAGEN = 28
CARPETA_DATASET = "mis_dibujos"

def cargar_datos():
    """Carga todas las imágenes"""
    imagenes = []
    etiquetas = []
    nombres_clases = []
    
    print("\n📂 Cargando tus dibujos...")
    
    # Obtener las clases
    for clase_nombre in sorted(os.listdir(CARPETA_DATASET)):
        ruta_clase = os.path.join(CARPETA_DATASET, clase_nombre)
        if os.path.isdir(ruta_clase):
            nombres_clases.append(clase_nombre)
            
            # Cargar cada imagen
            for archivo in os.listdir(ruta_clase):
                if archivo.endswith('.png'):
                    ruta_imagen = os.path.join(ruta_clase, archivo)
                    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Normalizar píxeles (0-1)
                        img = img.astype('float32') / 255.0
                        imagenes.append(img)
                        etiquetas.append(len(nombres_clases) - 1)
    
    print(f"✅ Cargadas {len(imagenes)} imágenes")
    print(f"📋 Clases encontradas: {nombres_clases}")
    
    return np.array(imagenes), np.array(etiquetas), nombres_clases

def crear_modelo(num_clases):
    """Crea la red neuronal"""
    modelo = keras.Sequential([
        # Capa de entrada
        layers.Input(shape=(TAMAÑO_IMAGEN, TAMAÑO_IMAGEN, 1)),
        
        # Capas convolucionales
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Capas densas
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_clases, activation='softmax')
    ])
    
    return modelo

# 1. Cargar datos
X, y, nombres_clases = cargar_datos()

if len(X) == 0:
    print("\n❌ ERROR: No se encontraron imágenes.")
    print("   Primero ejecuta 01_colector_dibujos.py para crear dibujos.")
    exit()

# 2. Preparar datos
X = X.reshape(-1, TAMAÑO_IMAGEN, TAMAÑO_IMAGEN, 1)

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Partición de datos:")
print(f"   Entrenamiento: {len(X_train)} imágenes")
print(f"   Prueba: {len(X_test)} imágenes")

# 3. Crear modelo
modelo = crear_modelo(len(nombres_clases))
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🧠 Resumen del modelo:")
modelo.summary()

# 4. Entrenar
print("\n🏋️  Comenzando entrenamiento...")
print("   (Esto puede tomar unos minutos)")
print("-" * 40)

historial = modelo.fit(
    X_train, y_train,
    epochs=15,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# 5. Evaluar
print("\n📈 Evaluando modelo...")
test_loss, test_acc = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 PRECISIÓN DEL MODELO: {test_acc*100:.2f}%")

# 6. Guardar modelo
modelo.save('modelo_dibujos.keras')
print(f"\n💾 Modelo guardado como 'modelo_dibujos.keras'")

# 7. Graficar resultados
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig('grafico_entrenamiento.png')
plt.show()

print("\n✨ ¡ENTRENAMIENTO COMPLETADO!")
print("   Ahora ejecuta 03_reconocedor_tiempo_real.py para probar tu IA")