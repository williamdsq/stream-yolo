from flask import Flask, Response
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)

# Carrega modelo YOLOv8 (leve)
model = YOLO('yolov8n.pt')

# Abre webcam (ou troque por caminho de vídeo se quiser)
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Processamento com YOLO
        results = model(frame)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Pessoa
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, 'Pessoa', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return 'YOLOv8 Stream está rodando! Acesse /video_feed para ver.'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # O Render define essa variável
    app.run(host='0.0.0.0', port=port)
