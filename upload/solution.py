import os
import numpy as np
from typing import List, Union, Tuple
from ultralytics import YOLO
import torch
import cv2

# Загружаем модель YOLOv11
model_path = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(model_path)

def split_image(image: np.ndarray, patch_size: int = 640) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    h, w = image.shape[:2]
    patches = []
    
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append((patch, (x, y)))
    
    return patches

def merge_boxes(boxes: List[dict], image_size: Tuple[int, int]) -> List[dict]:
    if not boxes:
        return []
    
    abs_boxes = []
    w_img, h_img = image_size
    
    for box in boxes:
        xc_abs = box['xc'] * w_img
        yc_abs = box['yc'] * h_img
        w_abs = box['w'] * w_img
        h_abs = box['h'] * h_img
        
        abs_boxes.append({
            'xc': xc_abs,
            'yc': yc_abs,
            'w': w_abs,
            'h': h_abs,
            'label': box['label'],
            'score': box['score']
        })
    
    # Простой алгоритм объединения пересекающихся боксов
    merged = []
    while abs_boxes:
        current = abs_boxes.pop(0)
        to_merge = [current]
        
        # Ищем пересекающиеся боксы
        i = 0
        while i < len(abs_boxes):
            box = abs_boxes[i]
            if boxes_intersect(current, box):
                to_merge.append(abs_boxes.pop(i))
            else:
                i += 1
        
        # Объединяем найденные боксы
        if len(to_merge) > 1:
            merged_box = combine_boxes(to_merge)
            merged.append(merged_box)
        else:
            merged.append(current)
    
    # Нормализуем обратно к [0,1]
    final_boxes = []
    for box in merged:
        final_boxes.append({
            'xc': box['xc'] / w_img,
            'yc': box['yc'] / h_img,
            'w': box['w'] / w_img,
            'h': box['h'] / h_img,
            'label': box['label'],
            'score': max(b['score'] for b in to_merge)  # Берем максимальную уверенность
        })
    
    return final_boxes

def boxes_intersect(box1: dict, box2: dict) -> bool:
    """Проверяет пересекаются ли два бокса"""
    x1_1 = box1['xc'] - box1['w']/2
    y1_1 = box1['yc'] - box1['h']/2
    x2_1 = box1['xc'] + box1['w']/2
    y2_1 = box1['yc'] + box1['h']/2
    
    x1_2 = box2['xc'] - box2['w']/2
    y1_2 = box2['yc'] - box2['h']/2
    x2_2 = box2['xc'] + box2['w']/2
    y2_2 = box2['yc'] + box2['h']/2
    
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

def combine_boxes(boxes: List[dict]) -> dict:
    x_coords = []
    y_coords = []
    scores = []
    
    for box in boxes:
        x1 = box['xc'] - box['w']/2
        x2 = box['xc'] + box['w']/2
        y1 = box['yc'] - box['h']/2
        y2 = box['yc'] + box['h']/2
        
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
        scores.append(box['score'])
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return {
        'xc': (x_min + x_max) / 2,
        'yc': (y_min + y_max) / 2,
        'w': x_max - x_min,
        'h': y_max - y_min,
        'label': boxes[0]['label'],
        'score': max(scores)  # Берем максимальную уверенность
    }

def infer_image_bbox(image: np.ndarray) -> List[dict]:
    h, w = image.shape[:2]
    patches = split_image(image)
    all_boxes = []
    
    for patch, (x_offset, y_offset) in patches:
        result = model.predict(source=patch, imgsz=640, device=0 if torch.cuda.is_available() else 'cpu')
        
        for res in result:
            for box in res.boxes:
                # Координаты относительно патча
                xc_patch = box.xywhn[0][0].item()
                yc_patch = box.xywhn[0][1].item()
                w_patch = box.xywhn[0][2].item()
                h_patch = box.xywhn[0][3].item()
                conf = box.conf[0].item()
                
                # Пересчитываем координаты относительно исходного изображения
                xc = (xc_patch * patch.shape[1] + x_offset) / w
                yc = (yc_patch * patch.shape[0] + y_offset) / h
                w_norm = w_patch * patch.shape[1] / w
                h_norm = h_patch * patch.shape[0] / h
                
                all_boxes.append({
                    'xc': xc,
                    'yc': yc,
                    'w': w_norm,
                    'h': h_norm,
                    'label': 0,  # Предполагаем класс 0 (человек)
                    'score': conf
                })
    
    # Объединяем пересекающиеся боксы
    merged_boxes = merge_boxes(all_boxes, (w, h))
    
    return merged_boxes

def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    results = []
    if isinstance(images, np.ndarray):
        images = [images]

    for image in images:
        image_results = infer_image_bbox(image)
        results.append(image_results)
    
    return results
