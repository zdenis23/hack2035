import os
import json
import random
import ray
from ray import tune
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from sklearn.model_selection import train_test_split

# ============== REPRO/SEED ==============
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# ====================== ПУТИ (используем абсолютные пути для Ray) ======================
dataset_path = os.path.abspath("./datatrain")     # ожидание структуры: images/, labels/
output_path  = os.path.abspath("./working")
os.makedirs(output_path, exist_ok=True)

images_dir = os.path.abspath(os.path.join(dataset_path, "images"))
labels_dir = os.path.abspath(os.path.join(dataset_path, "labels"))
assert os.path.isdir(images_dir), f"Not found: {images_dir}"
assert os.path.isdir(labels_dir), f"Not found: {labels_dir}"

# ====================== СБОР ФАЙЛОВ С ВАЛИДНЫМИ ЛЕЙБЛАМИ ======================
all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
all_images = [f for f in all_images if os.path.exists(os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt"))]
if len(all_images) == 0:
    raise RuntimeError("В датасете не найдено изображений с метками.")

rnd = random.Random(SEED)
rnd.shuffle(all_images)

# ====================== ПОДМНОЖЕСТВО ДЛЯ ТЮНИНГА (РОВНО 100 ИЛИ МЕНЬШЕ ПРИ НЕДОСТАТКЕ) ======================
subset_size = min(100, len(all_images))
subset_images = all_images[:subset_size]

# Разделение 80/20 для тюнинга (на подмножестве)
sub_train_imgs, sub_val_imgs = train_test_split(subset_images, test_size=0.2, random_state=SEED)

# Полный сплит для финального обучения
full_train_imgs, full_val_imgs = train_test_split(all_images, test_size=0.2, random_state=SEED)

# ====================== ЗАПИСЬ СПИСКОВ В .TXT (АБСОЛЮТНЫЕ ПУТИ) ======================
def write_list(txt_path, image_names):
    with open(txt_path, "w") as f:
        for name in image_names:
            f.write(os.path.join(images_dir, name) + "\n")

train_sub_txt  = os.path.abspath(os.path.join(output_path, "train_sub.txt"))
val_sub_txt    = os.path.abspath(os.path.join(output_path, "val_sub.txt"))
train_full_txt = os.path.abspath(os.path.join(output_path, "train_full.txt"))
val_full_txt   = os.path.abspath(os.path.join(output_path, "val_full.txt"))

write_list(train_sub_txt, sub_train_imgs)
write_list(val_sub_txt, sub_val_imgs)
write_list(train_full_txt, full_train_imgs)
write_list(val_full_txt, full_val_imgs)

# ====================== СОЗДАНИЕ YAML ДЛЯ ТЮНИНГА И ФИНАЛА (АБСОЛЮТНЫЕ ПУТИ) ======================
data_sub_yaml_path  = os.path.abspath(os.path.join(output_path, "data_sub.yaml"))
data_full_yaml_path = os.path.abspath(os.path.join(output_path, "data_full.yaml"))

with open(data_sub_yaml_path, "w") as f:
    f.write(f"train: {train_sub_txt}\nval: {val_sub_txt}\nnc: 1\nnames: ['human']\n")

with open(data_full_yaml_path, "w") as f:
    f.write(f"train: {train_full_txt}\nval: {val_full_txt}\nnc: 1\nnames: ['human']\n")

# ====================== ДОПУСТИМЫЕ КЛЮЧИ И ФИЛЬТРАЦИЯ ======================
cfg_allowed = set(vars(get_cfg()).keys())

def filter_params(d: dict) -> dict:
    return {k: v for k, v in d.items() if k in cfg_allowed}

def filter_space(space: dict) -> dict:
    filtered = {k: v for k, v in space.items() if k in cfg_allowed}
    dropped = sorted(set(space.keys()) - set(filtered.keys()))
    if dropped:
        print(f"[INFO] Dropped unsupported keys from search_space: {dropped}")
    return filtered

# ====================== ИНИЦИАЛИЗАЦИЯ RAY (корректная конфигурация GPU/CPU) ======================
def _detect_gpu_count():
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:
        return 0

if ray.is_initialized():
    ray.shutdown()

available_gpus = 3#_detect_gpu_count()
num_cpus = 7#max(1, os.cpu_count() or 1)
ray.init(num_gpus=4, num_cpus=7, ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)

device_arg = [0]#"cpu" if available_gpus == 0 else "0"

# ====================== ПОИСКОВОЕ ПРОСТРАНСТВО ======================
search_space_raw = {
    "optimizer": tune.choice(["AdamW", "SGD"]),
    "lr0": tune.loguniform(5e-5, 8e-3),
    "lrf": tune.uniform(0.1, 0.8),
    "momentum": tune.uniform(0.8, 0.98),
    "weight_decay": tune.loguniform(1e-6, 5e-3),

    "warmup_epochs": tune.uniform(0.0, 3.0),
    "warmup_momentum": tune.uniform(0.5, 0.95),
    "warmup_bias_lr": tune.loguniform(1e-4, 1e-1),

    "box": tune.uniform(5.0, 9.0),
    "cls": tune.uniform(0.4, 1.2),
    "fl_gamma": tune.uniform(1.0, 2.0),

    "hsv_h": tune.uniform(0.0, 0.3),
    "hsv_s": tune.uniform(0.5, 0.9),
    "hsv_v": tune.uniform(0.3, 0.8),
    "degrees": tune.uniform(0.0, 20.0),
    "translate": tune.uniform(0.05, 0.25),
    "scale": tune.uniform(0.4, 1.0),
    "shear": tune.uniform(0.0, 0.2),
    "perspective": tune.uniform(0.0, 0.0015),
    "flipud": tune.uniform(0.0, 0.5),
    "fliplr": tune.uniform(0.0, 0.7),
    "mosaic": tune.uniform(0.7, 1.0),

    "close_mosaic": tune.choice([10, 15, 20]),
}
search_space = filter_space(search_space_raw)

# ====================== ФИКС-ПАРАМЕТРЫ (ТЮНИНГ СТРОГО НА 100 ФОТО) ======================
fixed_params_raw = {
    "data": data_sub_yaml_path,  # абсолютный путь к YAML 100-изображений
    "epochs": 3,                 # быстрый тюнинг; grace_period <= epochs
    "imgsz": 1280,
    "batch": 8,
    "patience": 20,
    "cos_lr": True,
    "workers": 7,#min(4, max(1, num_cpus - 1)),
    "project": "satellite_detection",
    "name": "ray_tune_results_100imgs",
    "seed": SEED,
    "device": device_arg,
    "single_cls": True,
    "save": False,
    "plots": False,
}
fixed_params = filter_params(fixed_params_raw)

# ====================== ЗАПУСК ТЮНИНГА (Ray Tune интеграция Ultralytics) ======================
def tune_model():
    model = YOLO("yolo11m.pt")
    result_grid = model.tune(
        use_ray=True,
        data=fixed_params["data"],
        space=search_space,
        iterations=20,
        epochs=fixed_params["epochs"],
        imgsz=fixed_params["imgsz"],
        batch=fixed_params["batch"],
        patience=fixed_params["patience"],
        cos_lr=fixed_params["cos_lr"],
        workers=fixed_params["workers"],
        project=fixed_params["project"],
        name=fixed_params["name"],
        seed=fixed_params["seed"],
        device=fixed_params["device"],
        single_cls=fixed_params["single_cls"],
        save=fixed_params["save"],
        plots=fixed_params["plots"],
        grace_period=1,  # <= epochs
    )
    return result_grid

result_grid = tune_model()

# ====================== ЛУЧШАЯ КОНФИГ И ФИНАЛЬНОЕ ОБУЧЕНИЕ ======================
metric_key = "metrics/mAP50-95(B)"
try:
    best_result = result_grid.get_best_result(metric=metric_key, mode="max")
except Exception:
    best_result = result_grid.get_best_result()

best_config = best_result.config
print("Best tuned hyperparams (raw):", best_config)

best_hparams = {k: best_config[k] for k in search_space.keys() if k in best_config}
os.makedirs(os.path.abspath("ray_best"), exist_ok=True)
with open(os.path.abspath("ray_best/best_config.json"), "w") as f:
    json.dump(best_hparams, f, indent=2)

# ====================== ФИНАЛЬНОЕ ОБУЧЕНИЕ НА ПОЛНОМ ДАТАСЕТЕ ======================
final_model = YOLO("yolo11m.pt")
final_train_args_raw = {
    **best_hparams,
    "data": data_full_yaml_path,  # абсолютный путь к YAML полного датасета
    "epochs": 120,
    "patience": 25,
    "batch": 40,                   # увеличьте до 8 при достаточной VRAM
    "imgsz": 1280,
    "project": "satellite_detection",
    "name": "final_satellite_model_full",
    "seed": SEED,
    "device": [0,1,2,3],         # 'cpu' или '0'
    "single_cls": True,
    "save_json": True,
    "plots": True
}
final_train_args = filter_params(final_train_args_raw)
final_model.train(**final_train_args)

# ====================== ЭКСПОРТ И ВАЛИДАЦИЯ ======================
final_model.export(format="pt", imgsz=[1280, 1280], opset=12)
metrics = final_model.val(
    data=data_full_yaml_path,
    batch=16,
    imgsz=1280,
    conf=0.01,
    iou=0.5
)
print(f"mAP50-95 = {metrics.box.map:.4f}")

# Завершение Ray
ray.shutdown()