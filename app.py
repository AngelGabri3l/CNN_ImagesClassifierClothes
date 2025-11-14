import flask
from flask import request, jsonify, send_from_directory
import onnxruntime
from PIL import Image
import numpy as np
import io

# 1. Configuración del Modelo
# Asegúrate de que este archivo esté en el mismo directorio.
MODEL_PATH = 'cnn_fashion.onnx'
ort_session = onnxruntime.InferenceSession(MODEL_PATH)

# Nombres de las clases (ajusta esto según tu modelo)
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# La forma (shape) de entrada que espera tu modelo (ejemplo: 1, 1, 28, 28 para Fashion MNIST)
INPUT_SHAPE = (1, 1, 28, 28) 
INPUT_NAME = ort_session.get_inputs()[0].name

app = flask.Flask(__name__)

# Función de Preprocesamiento
def preprocess_image(image_bytes):
    # Cargar y convertir a escala de grises (si aplica)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    # Redimensionar al tamaño del modelo (28x28)
    image = image.resize(INPUT_SHAPE[2:]) 
    # Convertir a array de numpy y normalizar
    img_array = np.array(image, dtype=np.float32) / 255.0
    # Añadir las dimensiones batch y canal
    img_array = np.expand_dims(img_array, axis=(0, 1))
    return img_array

# 2. Ruta para servir el HTML (la vista estática)
@app.route('/')
def index():
    # Sirve el index.html desde el directorio raíz
    return send_from_directory('.', 'index.html')

# 3. Ruta para la Clasificación (la API)
@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró archivo de imagen'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        # Preprocesar la imagen
        input_data = preprocess_image(image_bytes)
        
        # Realizar la inferencia con ONNX Runtime
        ort_inputs = {INPUT_NAME: input_data}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Obtener la predicción
        prediction = np.argmax(ort_outs[0])
        result_class = CLASS_NAMES[prediction]
        
        return jsonify({'class': result_class, 'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ejecuta el servidor en modo debug solo para desarrollo local
    app.run(debug=True)