import numpy as np
import cv2
from keras.models import load_model
import time

cap = cv2.VideoCapture(0)

model = load_model('modelo.h5')

# Variáveis para cálculo de FPS
start_time = time.time()
num_frames = 0

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
        return '20 KM/H'
    elif classNo == 8:
        return 'PARE'

while True:
    success, imgOriginal = cap.read()

    # Converta o quadro para tons de cinza
    gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    # Aplique um desfoque para remover ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use o detector de bordas Canny para encontrar contornos
    edges = cv2.Canny(blurred, 50, 150)

    # Encontre contornos na imagem
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Itere sobre os contornos encontrados
    for contour in contours:
        # Calcule a área do contorno
        area = cv2.contourArea(contour)

        # Defina um limite para a área (ajuste conforme necessário)
        area_limite = 2000

        # Se a área do contorno for maior que o limite, processe
        if area > area_limite:
            # Aplique aproximação de polígono ao contorno
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Se o contorno possui 4 lados, considere como um possível retângulo (placa)
            if len(approx) == 8:
                # Extraia informações sobre a região retangular
                x, y, w, h = cv2.boundingRect(contour)

                # Desenhe o retângulo na imagem original
                cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extrair a região do retângulo
                rectangle_region = imgOriginal[y:y + h, x:x + w]

                # Redimensionar a região para 32x32
                rectangle_region_resized = cv2.resize(rectangle_region, (32, 32))

                # Pré-processamento da imagem
                img = preprocessing(rectangle_region_resized)
                img = img.reshape(1, 32, 32, 1)

                # Faça a previsão usando o modelo
                predictions = model.predict(img)
                indexVal = np.argmax(predictions)
                probabilityValue = np.amax(predictions)
                print(indexVal, probabilityValue)

                # Mostrar apenas se a probabilidade for maior ou igual a 80%
                if probabilityValue >= 0.98:
                    # Adicione o texto na imagem original
                    cv2.putText(imgOriginal, str(getClassName(indexVal)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    # Calcular e exibir FPS
    num_frames += 1
    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time

    # Adicione o texto do FPS à imagem original
    cv2.putText(imgOriginal, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Deteccao de placas", imgOriginal)
    cv2.waitKey(1)

# Liberação de recursos
cap.release()
cv2.destroyAllWindows()