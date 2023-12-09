import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import pytesseract

cap = cv2.VideoCapture(0)

model = load_model('modelo.h5')

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
    elif classNo == 1:
        return 'PARE'

# Certifique-se de configurar o caminho para o executável Tesseract se não estiver no PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Função para detectar texto usando Tesseract OCR
def detect_text(image):
    # Converta a imagem para o formato BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    text = pytesseract.image_to_string(img)
    return text

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

            # Se o contorno possui 8 lados, consideramos como um possível retângulo (placa)
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

                try:
                    # Chama a função para detectar texto na placa de trânsito
                    resultado = detect_text(rectangle_region)
                    # Exibe o resultado
                    print("Texto na placa de trânsito:", resultado)

                    # Faça a previsão usando o modelo
                    predictions = model.predict(img)
                    indexVal = np.argmax(predictions)
                    probabilityValue = np.amax(predictions)
                    print(indexVal, probabilityValue)

                    # Compara o texto reconhecido com as classes
                    if resultado.strip().upper() == getClassName(indexVal).strip():
                        # Se for igual, exiba na tela
                        print("Texto reconhecido é igual à classe. Exibindo na tela.")
                        cv2.putText(imgOriginal, str(getClassName(indexVal)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        # Se não for igual, não exiba na tela
                        print("Texto reconhecido não é igual à classe. Não exibindo na tela.")

                except Exception as e:
                    print("Erro ao detectar texto:", str(e))

    cv2.imshow("Deteccao de placas", imgOriginal)

    # Verifica se a tecla 'q' foi pressionada para encerrar o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura do vídeo ao encerrar
cap.release()
cv2.destroyAllWindows()