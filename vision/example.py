import cv2
import torch
import urllib.request
import os  # Import the os module
import warnings


# Download the model from the provided URL
model_url = "https://github.com/jhan15/traffic_cones_detection/raw/master/model/best.pt"
model_path = "best.pt"

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    print(f"Downloading model from {model_url}...")
    urllib.request.urlretrieve(model_url, model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(device)

# Função para calcular distância (como no exemplo anterior)
def calcular_distancia(cone_width_pixels, cone_real_width=0.3, focal_length=700):
    distance = (cone_real_width * focal_length) / cone_width_pixels
    return distance


warnings.filterwarnings("ignore", category=FutureWarning)

# Captura de vídeo da câmera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de objetos no frame usando o modelo customizado
    results = model(frame)  # Realiza a inferência no frame capturado

    # Obtém as detecções (resultados) no formato xyxy (caixas delimitadoras, etc.)
    detections = results.xyxy[0].cpu().numpy()  # Converte para NumPy para facilitar o processamento

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection

        # Supondo que a classe do cone de trânsito seja identificada pela classe '0' no modelo treinado
        if int(class_id) == 0:  # Ajuste conforme a classe correta do cone no seu modelo
            # Desenhar a caixa delimitadora ao redor do cone detectado
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Cone: {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Cálculo da largura do cone na imagem para estimar a distância
            cone_width = x2 - x1
            distance = calcular_distancia(cone_width)
            cv2.putText(frame, f"Distancia: {distance:.2f}m", (int(x1), int(y2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Cone detectado nas coordenadas: {(x1, y1, x2, y2)}")
            print(f"Distância estimada: {distance:.2f} metros")

    # Exibição do frame com as detecções
    cv2.imshow('Detecção', frame)

    # Tecla de saída
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
