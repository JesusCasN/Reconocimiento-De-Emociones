import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from PIL import ImageTk
import face_recognition
import shutil
import glob
from modelos.raza import *
from modelos.edad import *

class EmotionClassifierApp:
    def __init__(self, window, window_title, video_source=0):

        # Inicializar las propiedades de la ventana principal
        self.recognized_person_name = None  # Almacena el nombre de la persona reconocida (si la hay)
        self.window = window  # Ventana principal de la aplicación (raíz de Tkinter)
        self.window.title(window_title)  # Establece el título de la ventana
        self.video_source = video_source  # Fuente del video, por defecto es 0 (cámara web)

        # Dimensiones de la UI
        self.window_width = 350  # Ancho del área de visualización del video
        self.window_height = 400  # Alto del área de visualización del video
        self.ancho = 1024  # Ancho total de la ventana
        self.alto = 768  # Alto total de la ventana
        self.window.geometry(f'{self.ancho}x{self.alto}')  # geometría de la ventana

        # Directorio para guardar rostros reconocidos
        self.faces_dir = "archivos/saved_faces"
        self.setup_faces_directory()  # Prepara el directorio para guardar los rostros

        # Inicializar contador de rostros y máximo de rostros a reconocer
        self.face_count = 0  # Contador de rostros detectados
        self.max_faces = 30  # Máximo de rostros a detectar

        # Cargar modelos de análisis de emociones
        self.emotion_model = load_model('archivos/modelos_keras/modelo_clasificacion_emociones.keras', custom_objects={})

        #y detección de rostros
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Listas de emociones y colores asociados para la visualización
        self.emotions = ['disgustado', 'enojado', 'feliz', 'temeroso', 'neutral', 'sorprendido', 'triste']
        self.colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 255), (128, 128, 128), (255, 0, 0),
                       (255, 255, 0)]

        # Cargar modelo de género
        self.gender_net = cv2.dnn.readNetFromCaffe('archivos/genero/gender_deploy.prototxt', 'archivos/genero/gender_net.caffemodel')
        self.estimated_genders = []         # Lista para almacenar géneros estimados
        self.already_displayed_images = False  # Indicador si las imágenes ya fueron mostradas
        self.GENDER_LIST = ['Masculino', 'Femenino']

        # Cargar imágenes y codificaciones de ejemplo para reconocimiento
        self.rihanna_image = face_recognition.load_image_file("archivos/data/dataset_personas/rihanna.jpg")
        self.jungkook_image = face_recognition.load_image_file("archivos/data/dataset_personas/Jungkook.jpg")
        self.drake_image = face_recognition.load_image_file("archivos/data/dataset_personas/drake.jpeg")
        self.angelique_image = face_recognition.load_image_file("archivos/data/dataset_personas/Angelique.jpeg")
        self.eugenio_image = face_recognition.load_image_file("archivos/data/dataset_personas/Eugenio.jpeg")
        self.alfredo_image = face_recognition.load_image_file("archivos/data/dataset_personas/Alfredo.jpeg")
        self.eden_image = face_recognition.load_image_file("archivos/data/dataset_personas/Eden.jpeg")
        self.martha_image = face_recognition.load_image_file("archivos/data/dataset_personas/Martha.jpeg")
        self.rihanna_encoding = face_recognition.face_encodings(self.rihanna_image)[0]
        self.jungkook_encoding = face_recognition.face_encodings(self.jungkook_image)[0]
        self.drake_encoding = face_recognition.face_encodings(self.drake_image)[0]
        self.angelique_encoding = face_recognition.face_encodings(self.angelique_image)[0]
        self.eugenio_encoding = face_recognition.face_encodings(self.eugenio_image)[0]
        self.alfredo_encoding = face_recognition.face_encodings(self.alfredo_image)[0]
        self.eden_encoding = face_recognition.face_encodings(self.eden_image)[0]
        self.martha_encoding = face_recognition.face_encodings(self.martha_image)[0]

        # Configuración del video
        self.vid = cv2.VideoCapture(video_source)
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.window.configure(bg="#333333")  # Fondo gris oscuro
        self.canvas.configure(bg="#333333")  # Fondo de canvas oscuro
        self.canvas.pack()

        # Botones de control
        self.btn_loadvideo = tk.Button(window, text="Cargar Video", width=50, command=self.load_video, font=("Arial", 8), bg="#007BFF",
                                       fg="white", relief="flat")
        self.btn_loadvideo.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.btn_start = tk.Button(window, text="Iniciar", width=50, command=self.update, font=("Arial", 8), bg="#28a745", fg="white",
                                   relief="flat")
        self.btn_start.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.btn_start.bind("<Enter>",
                            lambda e: self.btn_start.configure(bg="#218838"))  # Cambia el color cuando el cursor entra
        self.btn_start.bind("<Leave>",
                            lambda e: self.btn_start.configure(bg="#28a745"))  # Vuelve al color original al salir

        # Iniciar el bucle principal de la interfaz
        self.window.mainloop()

    def setup_faces_directory(self):
        # Verifica si el directorio para guardar los rostros ya existe
        if os.path.exists(self.faces_dir):
            # Si el directorio existe, lo elimina. Esto es útil para asegurar que el directorio esté limpio
            # al inicio de cada sesión de la aplicación, evitando acumulación de imágenes de sesiones anteriores.
            shutil.rmtree(self.faces_dir)

        # Crea un nuevo directorio donde se guardarán los rostros detectados.
        # Esto es necesario ya que el directorio anterior fue eliminado o no existía.
        os.makedirs(self.faces_dir)

    def load_video(self):
        # Limpiar la interfaz de las imágenes de la persona anterior
        self.clear_person_images()

        # Abrir el cuadro de diálogo para seleccionar el archivo del video
        video_path = filedialog.askopenfilename()
        if video_path:
            # Reiniciar la captura de video con el nuevo archivo
            self.vid = cv2.VideoCapture(video_path)
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Restablecer el conteo de caras y la indicación de que se han mostrado las imágenes
            self.face_count = 0
            self.already_displayed_images = False

            # También limpiar las estimaciones anteriores de edad y género
            self.estimated_genders.clear()

            # Si ya se tiene un nombre de persona reconocida, restablecerlo también
            self.recognized_person_name = None

    def update(self):
        # Intenta leer un frame del video desde la fuente especificada en 'self.vid'
        ret, frame = self.vid.read()

        # Si se logra leer un frame correctamente, 'ret' será True
        if ret:
            # Llama a la función 'display_frame' para procesar y mostrar el frame actual
            self.display_frame(frame)
            # Utilizar 'after' para planificar la próxima llamada a 'update' después de 10 milisegundos
            # Esto crea un bucle que permite que la aplicación lea y procese el video en tiempo real.
            self.window.after(10, self.update)
        else:
            # Si no hay más frames para leer (por ejemplo, el video ha terminado), imprime un mensaje.
            # Esto marca el fin del análisis del video.
            print("Análisis completado.")

    def display_frame(self, frame):
        # Procesa el frame recibido utilizando la función 'process_frame'
        # que devuelve el frame procesado y un frame combinado para la visualización
        processed_frame, combined_frame = self.process_frame(frame)

        # Verifica si el frame combinado es una imagen en color de tres canales (BGR)
        if combined_frame.ndim == 3 and combined_frame.shape[2] == 3:
            # Convierte la imagen de BGR (formato de OpenCV) a RGB (formato de Tkinter)
            combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

        # Convierte la imagen de numpy array a un formato que Tkinter puede usar
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(combined_frame))

        # Muestra la imagen en el lienzo de Tkinter. Crea una imagen en la posición (0, 0)
        # con 'anchor=tk.NW' que significa que la esquina superior izquierda de la imagen estará en (0, 0)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def preprocess(self, image):
        # Esta función prepara la imagen para ser procesada por una red neuronal,
        # en este caso, utilizada para la detección de género.

        # 'blobFromImage' crea un blob (conjunto de imágenes preprocesadas) a partir de la imagen dada.
        # Este método realiza varias operaciones importantes para adaptar la imagen a los requisitos del modelo:
        # 1. Redimensiona la imagen a 227x227.
        # 2. Realiza la normalización de la imagen usando los valores medios especificados (78.426, 87.768, 114.895),
        #    que son típicos para modelos entrenados en grandes conjuntos de datos como ImageNet.
        # 3. 'swapRB=False' indica que no se debe intercambiar los canales azul y rojo. OpenCV carga imágenes en formato BGR,
        #    y si el modelo espera imágenes en RGB, esta opción debería ser True. En este caso, se mantiene en False,
        #    lo que sugiere que el modelo también utiliza el formato BGR.

        blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)

        # Retorna el blob, que es ahora un conjunto de datos listo para ser procesado por la red neuronal.
        return blob

    def print_most_common_gender(self):
        # Comprueba si la lista 'estimated_genders' contiene algún elemento
        if self.estimated_genders:
            # Utiliza la función 'max' para encontrar el género más común en la lista.
            # La función 'set' convierte la lista en un conjunto para eliminar duplicados,
            # y 'key=self.estimated_genders.count' indica a 'max' que use la frecuencia de cada género
            # como criterio para determinar el 'máximo'.
            most_common_gender = max(set(self.estimated_genders), key=self.estimated_genders.count)

            # Imprime el género más común encontrado
            print("El género de la persona identificada es:", most_common_gender)
        else:
            # Si no hay datos en 'estimated_genders', imprime un mensaje indicando que no se pueden
            # hacer estimaciones de género.
            print("No se detectaron suficientes datos para estimar el género.")

    def display_person_images(self, person_name):
        # Primero, se limpia las imágenes previamente mostradas para evitar duplicados en la interfaz
        self.clear_person_images()

        # Verifica si las imágenes de la persona específica ya se han mostrado para evitar repetición
        if not self.already_displayed_images:
            # Construye la ruta al directorio donde se almacenan las imágenes de la persona reconocida
            person_dir = os.path.join('archivos/img/', person_name)

            # Comprueba si el directorio existe para asegurar que hay imágenes para mostrar
            if os.path.exists(person_dir):
                # Encuentra todos los archivos .jpg en el directorio de la persona
                image_paths = []
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.jpge'):
                    image_paths.extend(glob.glob(os.path.join(person_dir, ext)))

                # Itera sobre cada imagen encontrada en el directorio
                for img_path in image_paths:
                    # Abre la imagen y la redimensiona a miniaturas de 150x150 para una visualización uniforme
                    img = Image.open(img_path)
                    img.thumbnail((150, 150), Image.Resampling.LANCZOS)

                    # Convierte la imagen a un formato que Tkinter pueda usar y mostrar
                    photo = ImageTk.PhotoImage(img)

                    # Crea una etiqueta en Tkinter para mostrar la imagen y la coloca en la ventana
                    img_label = tk.Label(self.window, image=photo)
                    img_label.image = photo  # Mantén una referencia a la imagen para evitar que se pierda
                    img_label.pack(side=tk.LEFT)  # Coloca la etiqueta al lado izquierdo de la ventana

                # Después de mostrar todas las imágenes, marca que las imágenes ya han sido mostradas
                self.already_displayed_images = True

    def clear_person_images(self):
        # Elimina todas las imágenes previas de la interfaz
        # Esto se hace para limpiar la interfaz antes de mostrar nuevas imágenes o al cerrar la aplicación.
        for widget in self.window.winfo_children():
            # Itera sobre todos los widgets (componentes) que son hijos de la ventana principal.
            if isinstance(widget, tk.Label):  # Comprueba si el widget es una etiqueta (Label).
                widget.destroy()  # Destruye el widget si es una etiqueta, eliminándolo de la interfaz.

        # Restablece la variable que indica si las imágenes ya fueron mostradas.
        # Esto permite que la función de mostrar imágenes pueda operar de nuevo si es necesario.
        self.already_displayed_images = False

    def process_frame(self, frame):
        # Redimensiona el frame al tamaño especificado para la ventana.
        frame = cv2.resize(frame, (self.window_width, self.window_height))

        # Convierte el frame a escala de grises para la detección de rostros.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta rostros en la imagen en escala de grises.
        # '1.1' es el factor de escala para la detección, '5' es el número mínimo de vecinos,
        # y 'minSize' es el tamaño mínimo del rostro a detectar.
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        # Prepara un canvas para dibujar barras de emociones.
        bar_canvas_width = 300
        bar_canvas = np.zeros((frame.shape[0], bar_canvas_width, 3), dtype=np.uint8)

        # Inicializa la variable para reconocer a la persona.
        recognized_person = "Desconocido"

        for (x, y, w, h) in faces:
            if self.face_count < self.max_faces:
                # Extrae la imagen del rostro y guarda una copia en el disco.
                face_img = frame[y:y + h, x:x + w]
                face_path = os.path.join(self.faces_dir, f"face_{self.face_count}.jpg")
                cv2.imwrite(face_path, face_img)
                self.face_count += 1  # Incrementa el contador de rostros guardados.

            # Dibuja un rectángulo alrededor del rostro detectado en el frame.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Prepara la imagen del rostro para codificación y comparación.
            face_image = frame[y:y + h, x:x + w]
            rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(rgb_face_image)

            if len(face_encoding) > 0:
                # Calcula la distancia a codificaciones conocidas para identificar a la persona.
                distances = face_recognition.face_distance([self.rihanna_encoding, self.jungkook_encoding, self.drake_encoding, self.angelique_encoding, self.eugenio_encoding, self.alfredo_encoding, self.eden_encoding, self.martha_encoding],
                                                           face_encoding[0])
                min_distance_index = np.argmin(distances)

                if min_distance_index == 0:
                    recognized_person = "Rihanna"
                elif min_distance_index == 1:
                    recognized_person = "Jungkook"
                elif min_distance_index == 2:
                    recognized_person = "Drake"
                elif min_distance_index == 3:
                    recognized_person = "Angelique"
                elif min_distance_index == 4:
                    recognized_person = "Eugenio"
                elif min_distance_index == 5:
                    recognized_person = "Alfredo"
                elif min_distance_index == 6:
                    recognized_person = "Eden"
                elif min_distance_index == 7:
                    recognized_person = "Martha"

                if not self.already_displayed_images:
                    if recognized_person in ["Rihanna", "Jungkook", "Drake", "Angelique", "Eugenio", "Alfredo", "Eden", "Martha"]:
                        self.display_person_images(recognized_person)

                # Preprocesa la imagen del rostro para detección de género.
                blob = self.preprocess(face_image)
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                estimated_gender = self.GENDER_LIST[gender_preds[0].argmax()]
                self.estimated_genders.append(estimated_gender)
                cv2.putText(frame, recognized_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Prepara y visualiza el análisis de emociones en el bar_canvas.
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)
            predictions = self.emotion_model.predict(roi)[0]
            bar_spacing = 5
            number_of_emotions = len(self.emotions)
            bar_width = (bar_canvas_width - (bar_spacing * (number_of_emotions + 1))) // number_of_emotions

            for i, (emotion, score) in enumerate(zip(self.emotions, predictions)):
                bar_height = int(score * frame.shape[0])
                bar_x = bar_spacing + i * (bar_width + bar_spacing)
                cv2.rectangle(bar_canvas, (bar_x, frame.shape[0] - bar_height), (bar_x + bar_width, frame.shape[0]), self.colors[i], cv2.FILLED)
                cv2.putText(bar_canvas, f"{emotion}: {score * 100:.2f}%", (bar_x + 5, frame.shape[0] - bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Combina el frame original con el bar_canvas y devuelve el frame combinado.
        combined_frame = np.hstack((frame, bar_canvas))
        if recognized_person != "Desconocido":
            self.recognized_person_name = recognized_person
        return frame, combined_frame


app = EmotionClassifierApp(tk.Tk(), "Detector de emociones con GUI")
if app.recognized_person_name:
    print("--------------------------------------------------------------")
    print("-----------------------Analisis Completo----------------------")
    print("Persona identificada:", app.recognized_person_name)
app.print_most_common_gender()

# Directorio que contiene todas las imágenes
directory = 'archivos/saved_faces'  # Cambia esto a la ruta de tu directorio

# Listas para almacenar las predicciones
predictions = []
ages = []

# Procesar todas las imágenes en el directorio para raza y edad
for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(directory, filename)

        # Predice la raza
        predicted_class = predict_image(image_path, model_ft, device, val_transforms)
        predictions.append(predicted_class)

        # Prepara la imagen y predice la edad
        blob = load_and_prepare_image(image_path)
        if blob is not None:
            age = predict_age(blob, age_net)
            ages.append(age)

# Determinar la raza más común
if predictions:
    most_common_race = max(set(predictions), key=predictions.count)
    print(f'La raza de la persona identificada es: {most_common_race}')
else:
    print('No se realizaron predicciones de raza.')

# Determinar la edad más común
if ages:
    most_common_age = max(set(ages), key=ages.count)
    print(f'El rango de edad de la persona identificada es de: {most_common_age}')
    print("--------------------------------------------------------------")
else:
    print('No se realizaron predicciones de edad.')
    print("--------------------------------------------------------------")
