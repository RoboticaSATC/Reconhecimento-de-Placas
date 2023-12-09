import numpy as np
import cv2
from keras.models import load_model
from picamera2 import Picamera2
import time

cap = Picamera2()
cap.configure(cap.create_preview_configuration(main={"format": 'XRGB8888', "size": (1024, 768)}))
cap.start()

model = load_model('modelo.h5')

#FPS
prev_time = 0

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    if classNo == 0:
        return '20km/h'
    elif classNo == 8:
        return 'PARE'

while True:
    imgOriginal = cap.capture_array()

    # Converte o quadro para tons de cinza
    gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    # Aplica um desfoque para remover ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # detector de bordas Canny para encontrar contornos
    edges = cv2.Canny(blurred, 50, 150)

    #contornos na imagem
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Contornos encontrados
    for contour in contours:
        # Calcula a área do contorno
        area = cv2.contourArea(contour)

        # Define um limite para a área (ajuste conforme necessário)
        area_limite = 2000

        # Se a área do contorno for maior que o limite, processe
        if area > area_limite:
            # Aplica aproximação de polígono ao contorno
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Se o contorno possui 4 lados, consideramos como um possível retângulo (placa)
            if len(approx) == 8:
                # Extrai informações sobre a região retangular
                x, y, w, h = cv2.boundingRect(contour)

                # Desenha o retângulo na imagem original
                cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extrai a região do retângulo
                rectangle_region = imgOriginal[y:y + h, x:x + w]

                # Redimensionar a região para 32x32
                rectangle_region_resized = cv2.resize(rectangle_region, (32, 32))

                # Pré-processamento da imagem
                img = preprocessing(rectangle_region_resized)
                img = img.reshape(1, 32, 32, 1)

                # Faz a previsão usando o modelo
                predictions = model.predict(img)
                indexVal = np.argmax(predictions)
                probabilityValue = np.amax(predictions)
                print(indexVal, probabilityValue)

                # Mostra apenas se a probabilidade for maior ou igual a 95%
                if probabilityValue >= 0.95:
                    # Adicione o texto na imagem original
                    cv2.putText(imgOriginal, str(getClassName(indexVal)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    # Adiciona isso antes de mostrar a imagem
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    fps = 1 / elapsed_time

    # Adiciona o FPS à imagem
    cv2.putText(imgOriginal, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Deteccao de placas", imgOriginal)
    cv2.waitKey(1)
