import sys
import cv2
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

class MicroalgaeClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Microalgae Classifier')
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setText("Cargue una imagen para clasificar las microalgas")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_load = QPushButton('Cargar Imagen', self)
        self.btn_load.clicked.connect(self.load_image)

        self.btn_classify = QPushButton('Clasificar', self)
        self.btn_classify.clicked.connect(self.classify_image)
        self.btn_classify.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_classify)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_path = None

    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.btn_classify.setEnabled(True)

    def classify_image(self):
        if not self.image_path:
            return

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h

            if 0.9 <= aspect_ratio <= 1.1:
                shape = "Esférica/Redonda"
                color = (0, 255, 0)  # Verde para Chlorophyta
            elif 1.5 <= aspect_ratio <= 3.0:
                shape = "Ovalada/Elíptica"
                color = (0, 0, 255)  # Rojo para diatomeas
            else:
                shape = "Filamentosa"
                color = (255, 0, 0)  # Azul para Oscillatoria

            cv2.drawContours(image, [contour], -1, color, 2)
            results.append({'x': x, 'y': y, 'shape': shape})

        # Guardar el archivo CSV con los resultados
        df = pd.DataFrame(results)
        df.to_csv('microalgae_classification.csv', index=False)

        # Guardar la imagen con las microalgas clasificadas
        output_image_path = 'classified_microalgae.png'
        cv2.imwrite(output_image_path, image)

        # Mostrar la imagen procesada en la interfaz
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        # Mostrar un mensaje de éxito
        self.label.setText(f"Clasificación completada. Imagen guardada como {output_image_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MicroalgaeClassifier()
    ex.show()
    sys.exit(app.exec())