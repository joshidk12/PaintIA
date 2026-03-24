"""
COLECTOR DE DIBUJOS PARA IA
Este programa te permite dibujar y guardar imágenes para entrenar tu IA
"""

import cv2
import numpy as np
import os

class ColectorDibujos:
    def __init__(self):
        # Crear carpeta para guardar los dibujos
        self.carpeta_dataset = "mis_dibujos"
        if not os.path.exists(self.carpeta_dataset):
            os.makedirs(self.carpeta_dataset)
        
        # Lista de objetos que vamos a dibujar (puedes modificarla)
        self.clases = ["circulo", "cuadrado", "triangulo", "casa"]
        
        # Crear carpeta para cada clase
        for clase in self.clases:
            ruta_clase = os.path.join(self.carpeta_dataset, clase)
            if not os.path.exists(ruta_clase):
                os.makedirs(ruta_clase)
                print(f"📁 Creada carpeta para: {clase}")
        
        self.clase_actual = 0
        self.contador_imagenes = 0
        self.dibujando = False
        self.punto_anterior = None
        
        # Crear el lienzo (fondo blanco)
        self.lienzo = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        print("\n🎨 COLECTOR DE DIBUJOS INICIADO")
        print("=" * 40)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Maneja los eventos del mouse"""
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
            
    def guardar_dibujo(self):
        """Guarda el dibujo actual"""
        if self.contador_imagenes < 30:  # 30 imágenes por clase
            nombre_archivo = f"{self.clases[self.clase_actual]}_{self.contador_imagenes}.png"
            ruta_completa = os.path.join(self.carpeta_dataset, self.clases[self.clase_actual], nombre_archivo)
            
            # Redimensionar a 28x28 píxeles
            img_pequena = cv2.resize(self.lienzo, (28, 28))
            cv2.imwrite(ruta_completa, img_pequena)
            
            print(f"✅ Guardado: {self.clases[self.clase_actual]}/{nombre_archivo}")
            self.contador_imagenes += 1
            
            # Limpiar el lienzo
            self.lienzo = np.ones((400, 400, 3), dtype=np.uint8) * 255
        else:
            print(f"⚠️ Ya tienes 30 imágenes de {self.clases[self.clase_actual]}. Cambia de clase.")
            
    def ejecutar(self):
        """Bucle principal"""
        cv2.namedWindow("Dibuja para tu IA")
        cv2.setMouseCallback("Dibuja para tu IA", self.mouse_callback)
        
        print("\n📝 INSTRUCCIONES:")
        print("  - Dibuja con el mouse (botón izquierdo)")
        print("  - Presiona 'g' para GUARDAR el dibujo")
        print("  - Presiona ESPACIO para cambiar de objeto")
        print("  - Presiona 'c' para LIMPIAR el lienzo")
        print("  - Presiona ESC para SALIR")
        print("=" * 40)
        print(f"🎯 Dibujando: {self.clases[self.clase_actual]}")
        print(f"📸 Imágenes: {self.contador_imagenes}/30")
        
        while True:
            # Mostrar información en la ventana
            visualizacion = self.lienzo.copy()
            
            # Agregar texto de instrucciones
            cv2.putText(visualizacion, f"Dibuja: {self.clases[self.clase_actual]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(visualizacion, f"Imagenes: {self.contador_imagenes}/30", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow("Dibuja para tu IA", visualizacion)
            tecla = cv2.waitKey(1) & 0xFF
            
            if tecla == ord('g'):
                self.guardar_dibujo()
                print(f"🎯 Dibujando: {self.clases[self.clase_actual]} ({self.contador_imagenes}/30)")
                
            elif tecla == ord(' '):
                self.clase_actual = (self.clase_actual + 1) % len(self.clases)
                self.contador_imagenes = len(os.listdir(os.path.join(self.carpeta_dataset, self.clases[self.clase_actual])))
                self.lienzo = np.ones((400, 400, 3), dtype=np.uint8) * 255
                print(f"\n🎯 Cambiando a: {self.clases[self.clase_actual]}")
                print(f"📸 Imágenes actuales: {self.contador_imagenes}/30")
                
            elif tecla == ord('c'):
                self.lienzo = np.ones((400, 400, 3), dtype=np.uint8) * 255
                print("🧹 Lienzo limpiado")
                
            elif tecla == 27:  # ESC
                print("\n📊 RESUMEN FINAL:")
                for clase in self.clases:
                    num = len(os.listdir(os.path.join(self.carpeta_dataset, clase)))
                    print(f"  - {clase}: {num} imágenes")
                break
                
        cv2.destroyAllWindows()

# Punto de entrada del programa
if __name__ == "__main__":
    colector = ColectorDibujos()
    colector.ejecutar()