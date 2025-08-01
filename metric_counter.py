import os
import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
# Импортируйте вашу функцию predict здесь
from solution import predict


# [ Параметры ]
EXTENSION = '.jpg'
COLUMNS = ['image_id', 'label', 'xc', 'yc', 'w', 'h', 'w_img', 'h_img', 'score', 'time_spent']

# Фиксируем сиды
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def process_images(images_path: str,
        result_csv_path: str = None) -> pd.DataFrame:
    """Функция обработки папки с изображениями, обрабатывает изображения по одному,
    отправляет их на функцию predict из файла solution.py (нужно переписать на свою),
    рассчитывает время инференса на изображение и записывает результат в pd.DataFrame
    
    Args:
        images_path (str): путь до директории с изображениями
        result_csv_path (str, optional): Путь, куда сохранять результат предсказаний. По-умолчанию None.

    Raises:
        Exception: Директория {images_dir} не найдена!
        Exception: Отсутствуют изображения в папке {images_dir}
        Exception: Время выполнения на тестовой выборке {time_spent} превышает {TIME_LIMIT_TEST // 60} минут!

    Returns:
        pd.DataFrame: датафрейм с результатом предсказаний
    """
    # Тестовая папка должна содержать подпапку images
    images_dir = os.path.join(images_path, 'images')
    image_paths = list(Path(images_dir).glob(f'*{EXTENSION}'))
    # Перемешиваем пути c изображениями
    random.shuffle(image_paths)
    images_count = len(os.listdir(images_dir))
    
    # Проверяем, существует ли директория, изображения
    assert os.path.exists(images_dir), Exception(f"Директория {images_dir} не найдена!")
    assert images_count > 0, Exception(f'Отсутствуют изображения в папке {images_dir}')

    results = []
    # Обрабатываем изображения по одному
    for image_path in image_paths:
        image_id = os.path.basename(image_path).split(EXTENSION)[0]
        # Открываем изображение в RGB формате
        image = cv2.imread(str(image_path), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = image.shape

        # Засекаем время выполнения функции predict
        start_time = time.time()
        # Вызываем функцию predict для одного изображения
        image_results = predict([image])
        # Останавливаем таймер
        elapsed_time = time.time() - start_time
        time_per_image = round(elapsed_time, 4)
        
        # Дополняем результаты ID изображения и затраченным временем
        if image_results and image_results[0]:
            for res in image_results[0]:
                res['image_id'] = image_id
                res['time_spent'] = time_per_image
                res['w_img'] = w_img
                res['h_img'] = h_img
                results.append(res)
        else:
            res = {'xc': None,
                   'yc': None,
                   'w': None,
                   'h': None,
                   'label': 0,
                   'score': None,
                   'image_id': image_id,
                   'time_spent': time_per_image,
                   'w_img': None,
                   'h_img': None
            }
            results.append(res)

    result_df = pd.DataFrame()
    if results and result_csv_path:
        result_df = pd.DataFrame(results, columns=COLUMNS)
        result_df = result_df.fillna(value=np.nan)
        result_df.to_csv(result_csv_path, index=False, na_rep=np.nan)
    print('Обработка выборки выполнена успешно!')
    
    return result_df


if __name__ == '__main__':
    assert len(sys.argv[1:]) == 2, Exception(f'Количество переданных аргументов: {len(sys.argv[1:])}, должно быть 2')
    if len(sys.argv[1:]) == 2:
        PRIVATE_DATASET_PATH, PRIVATE_CSV_PATH = sys.argv[1:]
        PRIVATE_DATASET_PATH = str(PRIVATE_DATASET_PATH)
        PRIVATE_CSV_PATH = str(PRIVATE_CSV_PATH)

    print(f'Инференс модели на private выборке')
    predicted_df = process_images(PRIVATE_DATASET_PATH,
                                  PRIVATE_CSV_PATH)