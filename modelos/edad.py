import cv2
import numpy as np
import os
from collections import Counter

def load_and_prepare_image(image_path):
    """ Carga una imagen y la prepara para la detección de edad. """
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return None
    image = cv2.resize(image, (227, 227))  # Tamaño esperado por la red de edad
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    return blob

def predict_age(blob, age_net):
    """ Predice la edad de la persona en la imagen. """
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_classes = ["(0-5)", "(6-19)", "(20-30)", "(30-35)", "(36-40)", "(41-50)", "(51-60)", "(61-100)"]
    age = age_classes[age_preds[0].argmax()]
    return age

# Cargar el modelo preentrenado para la estimación de edad
age_net = cv2.dnn.readNetFromCaffe('archivos/edad/deploy_age.prototxt', 'archivos/edad/age_net.caffemodel')






