"""
RECONOCEDOR DE DIBUJOS EN TIEMPO REAL
¡Dibuja y la IA adivinará qué es!
"""

import cv2
import numpy as np
import tensorflow as tf
import os

class ReconocedorDibujos:
    def __init__(self):
        print("🎨 INICIANDO RECONOCEDOR DE DIBUJOS")
        print("=" * 40)
        
        # Cargar el modelo
        ruta_modelo = 'modelo_dibujos.keras'
        if not os.path.exists(ruta_modelo):
            print(f"❌ ERROR: No se encuentra el modelo.")
            print("   Primero ejecuta 02_entrenar_modelo.py")
            exit()
            
        self.modelo = tf.keras.models.load_model(ruta_modelo)
        
        # Obtener nombres de las clases
        self.nombres_clases = sorted([d for d in os.listdir('mis_dibujos') 
                                      if os.path.isdir(os.path.join('mis_dibujos', d))])
        
        print(f"✅ Modelo cargado correctamente")
        print(f"🎯 Clases reconocidas: {', '.join(self.nombres_clases)}")
        
        # Configuración del dibujo
        self.dibujando = False
        self.punto_anterior = None
        
        # Crear el lienzo
        self.lienzo = np.ones((500, 500, 3), dtype=np.uint8) * 255
        
        # Para suavizar predicciones
        self.ultimas_predicciones = []
        
        print("\n📝 INSTRUCCIONES:")
        print("  - Dibuja con el mouse (botón izquierdo)")
        print("  - Presiona 'c' para LIMPIAR")
        print("  - Presiona ESC para SALIR")
        print("=" * 40)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Maneja el dibujo con el mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dibujando = True
            self.punto_anterior = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dibujando:
            if self.punto_anterior:
                cv2.line(self.lienzo, self.punto_anterior, (x, y), (0, 0, 0), 3)
            self.punto_anterior = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dibujando = False
            self.punto_anterior = None
    
    def predecir(self):
        """Analiza el dibujo"""
        # Preparar imagen
        gris = cv2.cvtColor(self.lienzo, cv2.COLOR_BGR2GRAY)
        invertido = cv2.bitwise_not(gris)
        pequeno = cv2.resize(invertido, (28, 28))
        pequeno = pequeno.astype('float32') / 255.0
        pequeno = pequeno.reshape(1, 28, 28, 1)
        
        # Predecir
        predicciones = self.modelo.predict(pequeno, verbose=0)[0]
        clase = np.argmax(predicciones)
        confianza = predicciones[clase]
        
        # Suavizar
        self.ultimas_predicciones.append(clase)
        if len(self.ultimas_predicciones) > 5:
            self.ultimas_predicciones.pop(0)
        
        if self.ultimas_predicciones:
            valores, conteos = np.unique(self.ultimas_predicciones, return_counts=True)
            clase_suave = valores[np.argmax(conteos)]
        else:
            clase_suave = clase
            
        return clase_suave, confianza, predicciones
    
    def ejecutar(self):
        """Bucle principal"""
        cv2.namedWindow("IA Reconocedora de Dibujos")
        cv2.setMouseCallback("IA Reconocedora de Dibujos", self.mouse_callback)
        
        while True:
            visualizacion = self.lienzo.copy()
            
            # Predecir
            clase, confianza, _ = self.predecir()
            
            # Mostrar resultados
            texto = f"IA DICE: {self.nombres_clases[clase]}"
            cv2.putText(visualizacion, texto, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            texto_conf = f"Confianza: {confianza*100:.1f}%"
            cv2.putText(visualizacion, texto_conf, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Barra de confianza
            cv2.rectangle(visualizacion, (10, 70), (10 + int(confianza*200), 80), 
                         (0, 255, 0), -1)
            
            # Instrucciones
            cv2.putText(visualizacion, "C: Limpiar | ESC: Salir", (300, 480), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.imshow("IA Reconocedora de Dibujos", visualizacion)
            
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('c'):
                self.lienzo = np.ones((500, 500, 3), dtype=np.uint8) * 255
                self.ultimas_predicciones = []
                print("🧹 Lienzo limpiado")
            elif tecla == 27:
                print("\n👋 ¡Hasta luego!")
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    reconocedor = ReconocedorDibujos()
    reconocedor.ejecutar()