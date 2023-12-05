# Dzuba Denis

import tensorflow as tf
import matplotlib.pyplot as plt
import neural_structured_learning as nsl
import numpy as np
import cv2
import os
import pickle

from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from statistics import mode
from keras.models import load_model
from keras.preprocessing import image as image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages are logged (default behavior)
                                            # 1 = INFO messages are not printed
                                            # 2 = INFO and WARNING messages are not printed
                                            # 3 = INFO, WARNING, and ERROR messages are not printed

app = Flask(__name__)  # Инициализация Flask-приложения

# Константы для обработки изображений
IMG_SHAPE_HEIGTH = 128  # Пример значения, замените на нужное
IMG_SHAPE_WIDTH = 128   # Пример значения, замените на нужное
IMG_TOTAL_SHAPE = IMG_SHAPE_HEIGTH * IMG_SHAPE_WIDTH

# Метки классов
class_labels = {
    0: 'Патология сетчатки отсутствует',
    1: 'Друзы (отложения на сетчатке)',
    2: 'Хороидальная неоваскуляризация',
    3: 'Макулярный отёк'}

# Загрузка многомерных моделей
model_knn_fit = pickle.load(open("./Web App/model_knn_fit.pkl", "rb"))
model_svc_fit = pickle.load(open("./Web App/model_svc_fit.pkl", "rb"))
pymodel_lor_fit = pickle.load(open("./Web App/pymodel_lor_fit.pkl", "rb"))
pymodel_rfc_fit = pickle.load(open("./Web App/pymodel_rfc_fit.pkl", "rb"))

# Создание директорий для скачивания
UPLOAD_FOLDER = 'Web App/static/uploads'  # Папка для загруженных файлов
RESULT_FOLDER = 'Web App/static/results'  # Папка для результатов обработки

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Создание директорий, если они не существуют
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Голосование
def weighted_vote(results, weights):
    vote_counts = {label: 0 for label in class_labels.values()}
    for model_name, vote in results.items():
        vote_counts[vote] += weights[model_name]
    return max(vote_counts, key=vote_counts.get)

# Подготовка изображений
def prepare_image(file_path, img_height, img_width):
    img = image.load_img(file_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем 4-ю размерность
    img_array /= 255.0  # Нормализация пикселей
    return img_array





# Главная страница с ссылками на различные функции
@app.route('/')
def home():
    return render_template('main.html')





# Страница параметризации
@app.route('/parameterization', methods=['GET', 'POST'])
def parameterization():
    result_filename = None  # Инициализация переменной для имени файла результата
    if request.method == 'POST':
        image_file = request.files['image']
        method = request.form['method']
        p = int(request.form.get('P', 8))  # Параметры для LBP
        r = int(request.form.get('R', 2))

        if image_file:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filepath = filepath.replace("\\", "/")  # Заменяем обратные слэши на прямые
            image_file.save(filepath)  # Сохранение файла

            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if method == 'hog':
                fd, hog_image = hog(image, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,)
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 15))
                result_filename = 'hog_result.jpg'
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                result_path = result_path.replace("\\", "/")  # Заменяем обратные слэши на прямые
                cv2.imwrite(result_path, hog_image_rescaled * 255)

            elif method == 'lbp':
                lbp_image = local_binary_pattern(image, P=p, R=r, method='uniform')
                result_filename = 'lbp_result.jpg'
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                result_path = result_path.replace("\\", "/")  # Заменяем обратные слэши на прямые
                cv2.imwrite(result_path, lbp_image * 255)
            print("Saved HOG image to:", filepath)  # Для HOG

    return render_template('parameterization.html', result_image=result_filename)





# Страница многомерные модели
@app.route('/multidimensional_models', methods=['GET', 'POST'])
def multidimensional_models():
    classification_results = {}
    vote_results = []
    weights = {
        'LogisticRegression': 0.524,
        'RandomForestClassifier': 0.670,
        'KNeighborsClassifier': 0.506,
        'SVC': 0.718
    }

    if request.method == 'POST':
        # Получаем файл из запроса
        imagefile = request.files['imagefile']
        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(image_path)

        # Обработка изображения
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_resized = cv2.resize(image, (IMG_SHAPE_WIDTH, IMG_SHAPE_HEIGTH)).flatten()
        image_for_model = image_resized.reshape(1, -1)  # Предобработка для модели

        # Словарь с моделями
        model_objects = {
            'RandomForestClassifier': pymodel_rfc_fit,
            'KNeighborsClassifier': model_knn_fit,
            'SVC': model_svc_fit,  # Убедитесь, что это SVC, а не SVR
            'LogisticRegression': pymodel_lor_fit
        }


        # Предсказание каждой моделью
        for model_name, model in model_objects.items():
            prediction = model.predict(image_for_model)
            result = class_labels[int(prediction[0])]
            classification_results[model_name] = result
            vote_results.append(result)

        overall_result = weighted_vote(classification_results, weights)

        return render_template('multidimensional_models.html',
                               classification_results=classification_results,
                               overall_result=overall_result,
                               image_show=filename)

    return render_template('multidimensional_models.html')





# Загрузите предварительно обученные модели (убедитесь, что указываете корректные пути к моделям)
IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'

base_model = tf.keras.models.load_model('CNN 4 Classes Generate (35%) 0001')

adv_model = tf.keras.models.load_model('NSL 4 Classes Generate (35%) 0001')
adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.025)
adv_model = nsl.keras.AdversarialRegularization(adv_model, label_keys=[LABEL_INPUT_NAME], adv_config=adv_config)
adv_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# Страница нейронные сети
@app.route('/neural_networks', methods=['GET', 'POST'])
def neural_networks():
    if request.method == 'POST':
        # Загрузка изображения
        imagefile = request.files['imagefile']
        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(image_path)
                
        img = Image.open(imagefile.stream).convert('RGB')  # Преобразуем в RGB если это необходимо
        
        # Обработка изображения для base_model
        img_base = img.resize((IMG_SHAPE_WIDTH, IMG_SHAPE_HEIGTH))
        img_array_base = np.array(img_base) / 255.0
        img_array_base = np.expand_dims(img_array_base, axis=0)  # добавляем измерение пакета

        # Делаем предсказание с базовой моделью
        predictions_base = base_model.predict(img_array_base)
        predicted_class_base = np.argmax(predictions_base, axis=1)

        # Делаем предсказание с adv моделью
        img_adv = img.resize((IMG_SHAPE_WIDTH, IMG_SHAPE_HEIGTH))
        img_array_adv = np.array(img_adv) / 255.0
        img_array_adv = np.expand_dims(img_array_adv, axis=0)  # добавляем измерение пакета
        img_array_adv = tf.convert_to_tensor(img_array_adv, dtype=tf.float32)
        input_dict = {"image": img_array_adv}
        label_placeholder = tf.convert_to_tensor([0], dtype=tf.int32)
        input_dict["label"] = label_placeholder

        predictions_adv = adv_model.predict(input_dict)
        predicted_class_adv = np.argmax(predictions_adv, axis=1)

        standard_prediction_label = class_labels[np.argmax(predictions_base)]
        nsl_prediction_label = class_labels[np.argmax(predictions_adv)]

        # Получаем предсказания от каждой модели
        standard_prediction_label = class_labels[predicted_class_base[0]]
        nsl_prediction_label = class_labels[predicted_class_adv[0]]

        # Голосование
        classification_results = {
            'StandardTraining': standard_prediction_label,
            'StructuredTraining': nsl_prediction_label
        }
        weights = {
            'StandardTraining': 0.954,
            'StructuredTraining': 0.965
        }
        overall_result = weighted_vote(classification_results, weights)

        # Подготовка результатов для передачи в шаблон
        return render_template('neural_networks.html', result_1=standard_prediction_label,
                               result_2=nsl_prediction_label, overall_result=overall_result, image_show=imagefile.filename)

    # Если GET запрос, просто отображаем страницу для загрузки изображения
    return render_template('neural_networks.html')




# Конец
if __name__ == '__main__':
    app.run(debug=True)