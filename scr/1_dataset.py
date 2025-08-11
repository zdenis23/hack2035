import os
import glob
import hashlib
import numpy as np
import polars as pl
from PIL import Image
from sahi.slicing import slice_coco
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import multiprocessing as mp
import shutil
from tqdm.auto import tqdm
from scipy.ndimage import laplace

# Разрешаем очень большие изображения
Image.MAX_IMAGE_PIXELS = None

# --- КОНФИГУРАЦИЯ ---
CONFIG = {
    "DATASETS_ROOT": "./dataset",
    "OUTPUT_DIR": "E://yolo_dataset",
    "BEST_IMAGES_COUNT": 20000,
    "CLASS_NAMES": ["human"],
    "SMALL_OBJECT_AREA_THRESHOLD": 32 * 32,
    "SHARPNESS_THRESHOLD": 1000,
    "SAHI_SLICE_SIZE": (1280, 1280),
    "SAHI_OVERLAP_RATIO": (0.1, 0.1),
    "ALB_AUGMENTATION_COUNT": 3,
    # Ограничение только для оценки резкости: изображение даунскейлится
    # до этого максимального размера, чтобы безопасно обрабатывать фото любого размера
    "SHARPNESS_DOWNSCALE_MAX": 2048,
}

# --- ВСПОМОГАТЕЛЬНОЕ ---

def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_files_in_dir(dir_path):
    """
    Возвращает список изображений и меток в директории (рекурсивно).
    Используем отдельную функцию для многопроцессорного обхода.
    """
    image_paths = []
    label_paths = []
    # Ищем основные расширения изображений
    for ext in ("*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(glob.glob(os.path.join(dir_path, '**', ext), recursive=True))
    label_paths = glob.glob(os.path.join(dir_path, '**', '*.txt'), recursive=True)
    return image_paths, label_paths

def get_file_paths_multiprocess(root_dir):
    """
    Собираем пары (image_path -> label_path) многопроцессорно, с прогрессом.
    """
    print("Этап 1: Сбор путей к файлам")
    all_subdirs = [x[0] for x in os.walk(root_dir)]

    path_map = {}
    with mp.Pool(mp.cpu_count()) as pool:
        image_paths_total = []
        label_paths_total = []
        for images, labels in tqdm(
            pool.imap_unordered(find_files_in_dir, all_subdirs),
            total=len(all_subdirs),
            desc="Обход директорий",
            dynamic_ncols=True
        ):
            image_paths_total.extend(images)
            label_paths_total.extend(labels)

    # Сопоставление по имени файла (без расширения)
    label_map = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths_total}

    for img_path in image_paths_total:
        name_wo_ext = os.path.splitext(os.path.basename(img_path))[0]
        if name_wo_ext in label_map:
            path_map[img_path] = label_map[name_wo_ext]

    # Доп. проверка структуры вида .../<...>/labels/*.txt рядом с images
    for img_path in image_paths_total:
        img_filename = os.path.basename(img_path).split('.')[0]
        label_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels'))
        label_path = os.path.join(label_dir, img_filename + '.txt')
        if os.path.exists(label_path):
            path_map[img_path] = label_path

    print(f"Найдено {len(path_map)} изображений с метками.")
    return path_map

def calculate_image_score(image_path, label_path):
    """
    Считает характеристики изображения: число bbox, долю мелких объектов,
    резкость (дисперсия Лапласиана). Возвращает словарь метрик.
    """
    try:
        # Проверяем существование файлов с подробной диагностикой
        if not os.path.exists(image_path):
            # print(f"ДИАГНОСТИКА: Изображение не найдено: {image_path}")
            return None
        if not os.path.exists(label_path):
            # print(f"ДИАГНОСТИКА: Файл меток не найден: {label_path}")
            return None

        # Проверяем размер файла меток
        if os.path.getsize(label_path) == 0:
            # print(f"ДИАГНОСТИКА: Пустой файл меток: {label_path}")
            return None

        # Без ограничений по размеру: читаем только метаданные и даунскейлим для оценки резкости
        with Image.open(image_path) as img_pil:
            width, height = img_pil.size
            # Готовим уменьшенную версию в градациях серого для оценки резкости
            img_gray = img_pil.convert('L')
            max_dim = CONFIG["SHARPNESS_DOWNSCALE_MAX"]
            # Сохраняет пропорции и ограничивает по наибольшей стороне
            img_gray.thumbnail((max_dim, max_dim), Image.LANCZOS)
            img_gray_np = np.array(img_gray)

        bbox_count = 0
        small_object_count = 0
        valid_lines = 0
        
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:  # Пропускаем пустые строки
                    continue
                    
                parts = line.split()
                if len(parts) < 5:
                    # print(f"ДИАГНОСТИКА: Некорректная строка в {label_path}: '{line}' (частей: {len(parts)})")
                    continue
                    
                try:
                    # class_id x y w h (нормализованные)
                    class_id, x_center, y_center, width_bbox, height_bbox = map(float, parts[:5])
                    valid_lines += 1
                    bbox_count += 1
                    
                    h, w = height, width
                    bbox_area = (width_bbox * w) * (height_bbox * h)
                    if bbox_area < CONFIG["SMALL_OBJECT_AREA_THRESHOLD"]:
                        small_object_count += 1
                        
                except ValueError as ve:
                    # print(f"ДИАГНОСТИКА: Ошибка парсинга строки в {label_path}: '{line}' - {ve}")
                    continue

            if bbox_count == 0:
                # print(f"ДИАГНОСТИКА: Нет валидных меток в файле: {label_path} (всего строк: {len(lines)}, валидных: {valid_lines})")
                return None

            small_object_ratio = small_object_count / bbox_count if bbox_count > 0 else 0.0

            # Используем SciPy для вычисления дисперсии Лапласиана на уменьшенном изображении
            sharpness = 0.0 if img_gray_np is None or img_gray_np.size == 0 else float(laplace(img_gray_np, output=np.float64).var())

            return {
                "image_path": image_path,
                "label_path": label_path,
                "bbox_count": bbox_count,
                "small_object_ratio": small_object_ratio,
                "sharpness": sharpness,
            }
    except (MemoryError) as e:
        # Реактивная обработка ошибок, связанных с памятью
        print(f"ДИАГНОСТИКА: Ошибка памяти при обработке {image_path}: {e}")
        return None
    except Exception as e:
        print(f"ДИАГНОСТИКА: Неизвестная ошибка при обработке {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def _calculate_image_score_star(args):
    return calculate_image_score(*args)

def select_best_images(image_scores_df: pl.DataFrame) -> pl.DataFrame:
    """
    Пропорционально распределяет BEST_IMAGES_COUNT между группами по bbox_count,
    отбирая лучшие изображения в каждой группе по качеству (резкость + доля мелких объектов).
    Файлы с пустыми метками (bbox_count = 0) исключаются.
    """
    # Файлы с bbox_count = 0 уже исключены в calculate_image_score (возвращает None)
    if len(image_scores_df) == 0:
        print("Нет изображений с метками для отбора.")
        return pl.DataFrame(schema={"image_path": pl.Utf8, "label_path": pl.Utf8})
    
    # 1. Анализ распределения по группам bbox_count
    group_stats = (
        image_scores_df
        .group_by('bbox_count')
        .agg([
            pl.len().alias('count')
        ])
        .sort('bbox_count')
    )
    
    print("\nРаспределение файлов по количеству меток:")
    total_files = 0
    for row in group_stats.iter_rows():
        bbox_count, count = row[0], row[1]  # bbox_count, count
        total_files += count
        print(f"  {bbox_count} меток: {count} файлов")
    
    if total_files == 0:
        print("Нет файлов для обработки.")
        return pl.DataFrame(schema={"image_path": pl.Utf8, "label_path": pl.Utf8})
    
    # 2. Вычисление пропорционального распределения
    target_counts = {}
    allocated_total = 0
    
    print(f"\nПропорциональное распределение {CONFIG['BEST_IMAGES_COUNT']} файлов:")
    
    # Первый проход: базовое распределение
    for row in group_stats.iter_rows():
        bbox_count, count = row[0], row[1]
        proportion = count / total_files
        target_count = int(CONFIG["BEST_IMAGES_COUNT"] * proportion)
        # Не можем взять больше, чем есть в группе
        target_count = min(target_count, count)
        target_counts[bbox_count] = target_count
        allocated_total += target_count
        print(f"  {bbox_count} меток: {target_count} файлов (из {count} доступных, {proportion:.1%})")
    
    # 3. Распределение остатка
    remaining = CONFIG["BEST_IMAGES_COUNT"] - allocated_total
    if remaining > 0:
        print(f"\nРаспределение остатка {remaining} файлов:")
        # Сортируем группы по убыванию bbox_count для приоритета групп с большим количеством меток
        sorted_groups = sorted(target_counts.items(), key=lambda x: x[0], reverse=True)
        
        for bbox_count, current_target in sorted_groups:
            if remaining <= 0:
                break
            # Находим максимально возможное количество для этой группы
            available_count = group_stats.filter(pl.col('bbox_count') == bbox_count)['count'][0]
            max_additional = available_count - current_target
            additional = min(remaining, max_additional)
            if additional > 0:
                target_counts[bbox_count] += additional
                remaining -= additional
                print(f"  {bbox_count} меток: +{additional} файлов (итого {target_counts[bbox_count]})")
    
    # 4. Отбор лучших в каждой группе
    selected_dfs = []
    
    for bbox_count, target_count in target_counts.items():
        if target_count <= 0:
            continue
            
        # Фильтруем группу
        group_df = image_scores_df.filter(pl.col('bbox_count') == bbox_count)
        
        if len(group_df) == 0:
            continue
        
        # Нормализация метрик внутри группы
        eps = 1e-9
        group_norm = group_df.with_columns([
            (((pl.col('small_object_ratio') - pl.col('small_object_ratio').min()) / 
              (pl.col('small_object_ratio').max() - pl.col('small_object_ratio').min() + eps))
             .fill_null(0).alias('small_object_ratio_norm')),
            (((pl.col('sharpness') - pl.col('sharpness').min()) / 
              (pl.col('sharpness').max() - pl.col('sharpness').min() + eps))
             .fill_null(0).alias('sharpness_norm')),
        ])
        
        # Вычисление качества: резкость + доля мелких объектов
        group_with_quality = group_norm.with_columns([
            (0.7 * pl.col('sharpness_norm') + 0.3 * pl.col('small_object_ratio_norm')).alias('quality_score')
        ])
        
        # Отбор лучших в группе
        best_in_group = (
            group_with_quality
            .sort('quality_score', descending=True)
            .head(target_count)
        )
        
        selected_dfs.append(best_in_group)
    
    # 5. Объединение результатов
    if not selected_dfs:
        print("Не удалось отобрать ни одного файла.")
        return pl.DataFrame(schema={"image_path": pl.Utf8, "label_path": pl.Utf8})
    
    result = pl.concat(selected_dfs)[['image_path', 'label_path']]
    
    print(f"\nИтого отобрано: {len(result)} файлов")
    return result


def safecopy(src, dst):
    """
    Безопасное копирование файла с возвращаемым статусом.
    """
    try:
        # Проверяем существование исходного файла
        if not os.path.exists(src):
            tqdm.write(f"ОШИБКА копирования: исходный файл не существует: {src}")
            return False
            
        # Создаем директорию назначения
        dst_dir = os.path.dirname(dst)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
            
        # Проверяем, что целевой файл не существует или отличается
        if os.path.exists(dst):
            if os.path.getsize(src) == os.path.getsize(dst):
                return True  # Файл уже скопирован
                
        shutil.copy2(src, dst)  # copy2 сохраняет метаданные
        
        # Проверяем успешность копирования
        if not os.path.exists(dst):
            tqdm.write(f"ОШИБКА копирования: файл не создан: {dst}")
            return False
            
        return True
    except PermissionError as e:
        tqdm.write(f"ОШИБКА копирования (доступ запрещен) {src} -> {dst}: {e}")
        return False
    except OSError as e:
        tqdm.write(f"ОШИБКА копирования (системная ошибка) {src} -> {dst}: {e}")
        return False
    except Exception as e:
        tqdm.write(f"ОШИБКА копирования (неизвестная) {src} -> {dst}: {e}")
        return False

def safemove(src, dst):
    """
    Безопасное перемещение файла с возвращаемым статусом.
    """
    try:
        # Проверяем существование исходного файла
        if not os.path.exists(src):
            tqdm.write(f"ОШИБКА перемещения: исходный файл не существует: {src}")
            return False
            
        # Создаем директорию назначения
        dst_dir = os.path.dirname(dst)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
            
        # Проверяем, что целевой файл не существует
        if os.path.exists(dst):
            if os.path.getsize(src) == os.path.getsize(dst):
                # Файл уже существует с тем же размером, удаляем исходный
                try:
                    os.remove(src)
                except:
                    pass
                return True
                
        shutil.move(src, dst)
        
        # Проверяем успешность перемещения
        if not os.path.exists(dst):
            tqdm.write(f"ОШИБКА перемещения: файл не создан: {dst}")
            return False
        if os.path.exists(src):
            tqdm.write(f"ОШИБКА перемещения: исходный файл не удален: {src}")
            return False
            
        return True
    except PermissionError as e:
        tqdm.write(f"ОШИБКА перемещения (доступ запрещен) {src} -> {dst}: {e}")
        return False
    except OSError as e:
        tqdm.write(f"ОШИБКА перемещения (системная ошибка) {src} -> {dst}: {e}")
        return False
    except Exception as e:
        tqdm.write(f"ОШИБКА перемещения (неизвестная) {src} -> {dst}: {e}")
        return False

def is_in_datasets_root(file_path):
    """
    Проверяет, находится ли файл в DATASETS_ROOT директории.
    """
    try:
        datasets_root = os.path.abspath(CONFIG["DATASETS_ROOT"])
        file_abs_path = os.path.abspath(file_path)
        return file_abs_path.startswith(datasets_root)
    except:
        return False

def safe_find_and_copy(original_path, dst, search_dirs=None, file_type="файл"):
    """
    Интеллектуальный поиск и копирование файла.
    Аналог safe_find_and_move, но только копирует файлы.
    """
    # Нормализуем пути для Windows
    original_path = os.path.normpath(original_path)
    dst = os.path.normpath(dst)
    
    # Проверяем, не существует ли уже целевой файл
    if os.path.exists(dst):
        return True  # Файл уже на месте
    
    # Проверяем исходный путь
    if os.path.exists(original_path):
        return safecopy(original_path, dst)
    
    # Если файл не найден по исходному пути, ищем по basename
    filename = os.path.basename(original_path)
    
    if search_dirs is None:
        search_dirs = [
            CONFIG["DATASETS_ROOT"],
            CONFIG["OUTPUT_DIR"],
            os.path.join(CONFIG["OUTPUT_DIR"], "sliced_yolo_temp"),
            os.path.join(CONFIG["OUTPUT_DIR"], "sliced_yolo_temp", "images"),
            os.path.join(CONFIG["OUTPUT_DIR"], "sliced_yolo_temp", "labels"),
            os.path.join(CONFIG["OUTPUT_DIR"], "sahi_src", "images"),
            "./temp_sliced_yolo",
            "./temp_sliced_yolo/images",
            "./temp_sliced_yolo/labels",
            "./temp_sliced_yolo/sahi_src_images"
        ]
        
        # Добавляем все поддиректории из DATASETS_ROOT
        if os.path.exists(CONFIG["DATASETS_ROOT"]):
            for root, dirs, files in os.walk(CONFIG["DATASETS_ROOT"]):
                search_dirs.append(root)
    
    # Ищем файл в указанных директориях
    found_paths = []
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        # Прямой поиск в директории
        direct_path = os.path.join(search_dir, filename)
        if os.path.exists(direct_path):
            if _is_image_file(direct_path) or direct_path.endswith('.txt'):
                found_paths.append(direct_path)
                
        # Рекурсивный поиск файла
        try:
            for root, dirs, files in os.walk(search_dir):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    if _is_image_file(found_path) or found_path.endswith('.txt'):
                        found_paths.append(found_path)
        except (OSError, PermissionError):
            continue
    
    # Убираем дубликаты и сортируем по длине пути (предпочитаем более короткие пути)
    found_paths = sorted(list(set(found_paths)), key=len)
    
    # Поиск без учета регистра (для Windows)
    if not found_paths:
        filename_lower = filename.lower()
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            try:
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.lower() == filename_lower:
                            found_path = os.path.join(root, file)
                            if _is_image_file(found_path) or found_path.endswith('.txt'):
                                found_paths.append(found_path)
            except (OSError, PermissionError):
                continue
    
    if found_paths:
        # Используем первый найденный файл
        src_path = found_paths[0]
        return safecopy(src_path, dst)
    else:
        tqdm.write(f"ПРЕДУПРЕЖДЕНИЕ: {file_type} не найден: {filename}")
        return False

def safe_transfer_file(src_path, dst_path, file_type="файл"):
    """
    Универсальная функция для безопасного переноса файла.
    Копирует файлы из DATASETS_ROOT, перемещает файлы из других мест.
    """
    if is_in_datasets_root(src_path):
        return safecopy(src_path, dst_path)
    else:
        return safemove(src_path, dst_path)

def safe_find_and_move(original_path, dst, search_dirs=None, file_type="файл"):
    """
    Интеллектуальный поиск и перемещение файла.
    Сначала проверяет исходный путь, затем ищет по basename в указанных директориях.
    ВАЖНО: НЕ включает DATASETS_ROOT для предотвращения перемещения из неизменяемой директории.
    """
    # Нормализуем пути для Windows
    original_path = os.path.normpath(original_path)
    dst = os.path.normpath(dst)
    
    # Проверяем, не существует ли уже целевой файл
    if os.path.exists(dst):
        return True  # Файл уже на месте
    
    # Проверяем исходный путь
    if os.path.exists(original_path):
        # КРИТИЧЕСКИ ВАЖНО: Проверяем, не находится ли файл в DATASETS_ROOT
        if is_in_datasets_root(original_path):
            return safecopy(original_path, dst)  # Копируем, не перемещаем
        else:
            return safemove(original_path, dst)  # Перемещаем только из других мест
    
    # Если файл не найден по исходному пути, ищем по basename
    filename = os.path.basename(original_path)
    
    if search_dirs is None:
        # ИСПРАВЛЕНО: Убираем DATASETS_ROOT из поиска для перемещения
        search_dirs = [
            CONFIG["OUTPUT_DIR"],
            os.path.join(CONFIG["OUTPUT_DIR"], "sliced_yolo_temp"),
            os.path.join(CONFIG["OUTPUT_DIR"], "sliced_yolo_temp", "images"),
            os.path.join(CONFIG["OUTPUT_DIR"], "sliced_yolo_temp", "labels"),
            os.path.join(CONFIG["OUTPUT_DIR"], "sahi_src", "images"),
            "./temp_sliced_yolo",
            "./temp_sliced_yolo/images",
            "./temp_sliced_yolo/labels",
            "./temp_sliced_yolo/sahi_src_images"
        ]
    
    # Ищем файл в указанных директориях
    found_paths = []
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        # Прямой поиск в директории
        direct_path = os.path.join(search_dir, filename)
        if os.path.exists(direct_path):
            if _is_image_file(direct_path) or direct_path.endswith('.txt'):
                found_paths.append(direct_path)
                
        # Рекурсивный поиск файла
        try:
            for root, dirs, files in os.walk(search_dir):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    if _is_image_file(found_path) or found_path.endswith('.txt'):
                        found_paths.append(found_path)
        except (OSError, PermissionError):
            continue  # Пропускаем недоступные директории
    
    # Удаляем дубликаты
    found_paths = list(set(found_paths))
    
    # Если найдено несколько копий, используем первую
    if found_paths:
        chosen_path = found_paths[0]
        if len(found_paths) > 1:
            tqdm.write(f"Найдено {len(found_paths)} копий {file_type} {filename}, используем: {chosen_path}")
        
        # КРИТИЧЕСКИ ВАЖНО: Проверяем, не находится ли найденный файл в DATASETS_ROOT
        if is_in_datasets_root(chosen_path):
            return safecopy(chosen_path, dst)  # Копируем, не перемещаем
        else:
            return safemove(chosen_path, dst)  # Перемещаем только из других мест
    
    # Если файл так и не найден, попробуем поиск без учета регистра (для Windows)
    filename_lower = filename.lower()
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        try:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.lower() == filename_lower:
                        found_path = os.path.join(root, file)
                        if _is_image_file(found_path) or found_path.endswith('.txt'):
                            tqdm.write(f"Найден {file_type} {filename} (без учета регистра) в {found_path}")
                            # КРИТИЧЕСКИ ВАЖНО: Проверяем, не находится ли найденный файл в DATASETS_ROOT
                            if is_in_datasets_root(found_path):
                                return safecopy(found_path, dst)  # Копируем, не перемещаем
                            else:
                                return safemove(found_path, dst)  # Перемещаем только из других мест
        except (OSError, PermissionError):
            continue
    
    # Дополнительная диагностика: показываем что есть в директориях
    existing_dirs = [d for d in search_dirs if os.path.exists(d)]
    if len(existing_dirs) > 0:
        tqdm.write(f"ДИАГНОСТИКА: Поиск {filename} в {len(existing_dirs)} директориях:")
        for search_dir in existing_dirs[:3]:  # Показываем первые 3 директории
            try:
                files_in_dir = []
                for root, dirs, files in os.walk(search_dir):
                    for f in files:
                        full_path = os.path.join(root, f)
                        if _is_image_file(full_path) or f.endswith('.txt'):
                            files_in_dir.append(f)
                        if len(files_in_dir) > 10:  # Ограничиваем вывод
                            break
                    if len(files_in_dir) > 10:
                        break
                tqdm.write(f"  {search_dir}: {len(files_in_dir)} файлов, примеры: {files_in_dir[:5]}")
            except (OSError, PermissionError):
                tqdm.write(f"  {search_dir}: недоступна")
    
    # Если файл так и не найден
    tqdm.write(f"ОШИБКА: {file_type} {filename} не найден ни в исходном пути {original_path}, ни в {len(existing_dirs)} поисковых директориях")
    return False

def _pick_best_coco_json(sliced_dir: str):
    """
    Находит лучший COCO JSON в каталоге слайсов,
    приоритет отдается sliced_coco.json. Возвращает путь или None.
    """
    json_candidates = glob.glob(os.path.join(sliced_dir, '**', '*.json'), recursive=True)
    
    if not json_candidates:
        return None
    
    # Приоритет для sliced_coco.json
    sliced_coco_candidates = [p for p in json_candidates if os.path.basename(p) == 'sliced_coco.json']
    if sliced_coco_candidates:
        return sliced_coco_candidates[0]
    
    # Исключаем source_coco.json как резерв
    json_candidates = [p for p in json_candidates if os.path.basename(p) != 'source_coco.json']
    
    if not json_candidates:
        return None
    
    best_path = None
    best_count = -1
    import json
    for cand in json_candidates:
        try:
            with open(cand, 'r', encoding='utf-8') as f:
                data = json.load(f)
            count = len(data.get('images', []))
            if count > best_count:
                best_count = count
                best_path = cand
        except Exception as e:
            continue
    return best_path

def ensure_yolo_labels_from_coco(sliced_dir: str, coco_json_path=None, image_dims_override=None) -> int:
    """
    Создаёт YOLO-метки (.txt) в sliced_dir/labels на основе COCO JSON,
    сгенерированного SAHI slice_coco. Возвращает число созданных файлов .txt.
    """
    # Выбираем COCO json: либо явный, либо ищем в стандартных местах
    if coco_json_path is None:
        coco_json_path = _pick_best_coco_json(sliced_dir)
        if coco_json_path is None:
            print("Не найден COCO JSON после нарезки SAHI; пропускаем конвертацию в YOLO.")
            return 0
    try:
        import json
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            coco_dict = json.load(f)
    except Exception as e:
        print(f"Не удалось прочитать COCO JSON '{coco_json_path}': {e}")
        return 0

    images = {img['id']: img for img in coco_dict.get('images', [])}
    anns_per_image = {}
    for ann in coco_dict.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id is None or img_id not in images:
            continue
        anns_per_image.setdefault(img_id, []).append(ann)

    labels_dir = os.path.join(sliced_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    created = 0
    # Готовим переопределения размеров (если были паддинги)
    dims_override = image_dims_override or {}

    for img_id, img_info in images.items():
        file_name = os.path.basename(img_info.get('file_name', ''))
        ow = float(img_info.get('width', 0) or 0)
        oh = float(img_info.get('height', 0) or 0)
        width, height = ow, oh
        if file_name in dims_override:
            width, height = dims_override[file_name]
        width = float(width or 0)
        height = float(height or 0)
        if width <= 0 or height <= 0:
            continue
        if not file_name:
            continue
        stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")

        lines = []
        for ann in anns_per_image.get(img_id, []):
            cat_id = int(ann.get('category_id', 0))
            x_min, y_min, w_box, h_box = ann.get('bbox', [0, 0, 0, 0])
            # Перевод в YOLO: нормализованные x_center, y_center, w, h
            x_center = (x_min + w_box / 2.0) / width
            y_center = (y_min + h_box / 2.0) / height
            w_norm = w_box / width
            h_norm = h_box / height
            # Ограничиваем в [0,1]
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)
            lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if lines:
            try:
                with open(label_path, 'w', encoding='utf-8') as lf:
                    lf.write("\n".join(lines) + "\n")
                created += 1
            except Exception as e:
                tqdm.write(f"Не удалось записать метку {label_path}: {e}")

    print(f"Сконвертировано в YOLO меток: {created}")
    return created

def ensure_sliced_images_folder(sliced_dir: str, search_roots=None, coco_json_path=None) -> int:
    """
    Гарантирует наличие директории sliced_dir/images и перемещает туда все изображения,
    перечисленные в COCO JSON, найденные рекурсивно в sliced_dir. Возвращает число
    успешно размещённых файлов в images/.
    """
    # Находим COCO JSON, созданный после slice_coco
    if coco_json_path is None:
        coco_json_path = _pick_best_coco_json(sliced_dir)
        if coco_json_path is None:
            return 0
    try:
        import json
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            coco_dict = json.load(f)
    except Exception:
        return 0

    images_info = coco_dict.get('images', [])
    images_dir = os.path.join(sliced_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    placed = 0
    # Индексируем все файлы по basename для быстрого поиска
    if search_roots is None:
        search_roots = [sliced_dir]
    all_files = []
    for root in search_roots:
        all_files.extend(glob.glob(os.path.join(root, '**', '*'), recursive=True))
    name_to_path = {}
    for p in all_files:
        if os.path.isfile(p):
            name_to_path.setdefault(os.path.basename(p), p)

    for img in images_info:
        rel_path = img.get('file_name', '') or ''
        file_name = os.path.basename(rel_path)
        if not file_name:
            continue
        dst = os.path.join(images_dir, file_name)
        if os.path.exists(dst):
            placed += 1
            continue
        # 1) Пытаемся найти точный относительный путь под каждым root
        found_src = None
        for root in search_roots:
   