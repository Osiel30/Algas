import sys
import cv2
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Worker(QThread):
    finished = pyqtSignal()  #proceso terminado
    result_ready = pyqtSignal(np.ndarray, list)  # Enviar imagen y csv

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        #Procesamiento
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Umbralizacion
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        features = []
        labels = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            #Color
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(image, mask=mask)

            #Textura
            contrast, energy = self.calculate_texture(gray)

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
            results.append({
                'x': x,
                'y': y,
                'shape': shape,
                'color': mean_color[:3], 
                'contrast': contrast,
                'energy': energy,
                'area': area,
                'perimeter': perimeter
            })

            #Se pasan los datos al Clasificador de Bayes
            features.append([mean_color[0], mean_color[1], mean_color[2], contrast, energy, area, perimeter])
            labels.append(shape)

        #Guarda el csv
        df = pd.DataFrame(results)
        df.to_csv('microalgae_classification.csv', index=False)

        #Evaluacion y entrenamiento
        if len(features) > 0 and len(labels) > 0:
            self.train_and_evaluate_bayes(features, labels)

        #resultados
        self.result_ready.emit(image, results)
        self.finished.emit()

    def calculate_texture(self, gray_image):
        gray_image = img_as_ubyte(gray_image)
        glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        return contrast, energy

    def train_and_evaluate_bayes(self, features, labels):
        #Datod de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        #Entrenamiento
        model = GaussianNB()
        model.fit(X_train, y_train)

        #Calcular precision
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        
        print("------------------ CLASIFICADOR NAIVE BAYES ------------------")
        print("Proceso de clasificación:")
        print(f"- Total de muestras: {len(features)}")
        print(f"- Muestras de entrenamiento: {len(X_train)}")
        print(f"- Muestras de prueba: {len(X_test)}")
        print(f"- Precisión (accuracy): {accuracy * 100:.2f}%")
        print("-------------------------------------------------------------")


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
        self.worker = None

    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.btn_classify.setEnabled(True)

    def classify_image(self):
        if not self.image_path:
            return

    
        self.btn_classify.setEnabled(False)

    
        self.worker = Worker(self.image_path)
        self.worker.result_ready.connect(self.update_results)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_results(self, image, results):
        #Guardar la imagen con las microalgas ya clasificadas
        output_image_path = 'classified_microalgae.png'
        cv2.imwrite(output_image_path, image)

        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        self.label.setText(f"Clasificación completada. Imagen guardada como {output_image_path}")

    def on_finished(self):
        self.btn_classify.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MicroalgaeClassifier()
    ex.show()
    sys.exit(app.exec())