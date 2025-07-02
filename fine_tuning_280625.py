import os
import json
import logging
import torch
import shutil
import numpy as np
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from train_model import setup_cfg, CustomTrainer, split_dataset

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_strong_augmentation(cfg):
    """
    Создает сильные аугментации для обучения
    """
    return T.AugmentationList([
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomSaturation(0.8, 1.2),
        # T.RandomLighting(0.7),  # Убрал - может не существовать в стандартной версии
        # T.RandomRotation([-15, 15]),  # Убрал - может не существовать в стандартной версии
        T.RandomFlip(horizontal=True, vertical=False),
        T.RandomFlip(horizontal=False, vertical=True),
        T.RandomCrop("relative_range", [0.8, 0.8]),
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        ),
    ])

class AugmentedTrainer(CustomTrainer):
    """
    Расширенный тренер с дополнительными аугментациями
    """
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def create_validation_dataset_from_best_datasets():
    """
    Создает валидационный датасет только из dataset_180625 и dataset_280625
    """
    val_annotations = {
        "licenses": [],
        "info": {"description": "Validation dataset from best datasets only"},
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # Загружаем аннотации из dataset_180625
    dataset_180625_annotations = "dataset_180625/annotations/instances_default.json"
    if os.path.exists(dataset_180625_annotations):
        with open(dataset_180625_annotations, 'r', encoding='utf-8') as f:
            data_180625 = json.load(f)
            logging.info(f"Загружено {len(data_180625['images'])} изображений из dataset_180625")
            
            # Добавляем категории (если ещё не добавлены)
            if not val_annotations['categories']:
                val_annotations['categories'] = data_180625['categories']
            
            # Добавляем изображения с префиксом
            for img in data_180625['images']:
                img['file_name'] = f"dataset_180625_{img['file_name']}"
                val_annotations['images'].append(img)
            
            # Добавляем аннотации с обновленными image_id
            for ann in data_180625['annotations']:
                val_annotations['annotations'].append(ann)
    
    # Загружаем аннотации из dataset_280625
    dataset_280625_annotations = "dataset_280625/annotations/instances_default.json"
    if os.path.exists(dataset_280625_annotations):
        with open(dataset_280625_annotations, 'r', encoding='utf-8') as f:
            data_280625 = json.load(f)
            logging.info(f"Загружено {len(data_280625['images'])} изображений из dataset_280625")
            
            # Добавляем изображения с префиксом
            for img in data_280625['images']:
                img['file_name'] = f"dataset_280625_{img['file_name']}"
                val_annotations['images'].append(img)
            
            # Добавляем аннотации с обновленными image_id
            for ann in data_280625['annotations']:
                val_annotations['annotations'].append(ann)
    
    logging.info(f"Создан валидационный датасет: {len(val_annotations['images'])} изображений, {len(val_annotations['annotations'])} аннотаций")
    return val_annotations

def merge_two_datasets(ds1_dir, ds2_dir, out_json_path, out_images_dir):
    """
    Объединяет два COCO-датасета (аннотации и изображения) в один.
    """
    os.makedirs(out_images_dir, exist_ok=True)
    merged = {
        "licenses": [],
        "info": {"description": "Merged dataset of two latest datasets"},
        "categories": [],
        "images": [],
        "annotations": []
    }
    ann_files = [
        os.path.join(ds1_dir, "annotations", "instances_default.json"),
        os.path.join(ds2_dir, "annotations", "instances_default.json")
    ]
    img_dirs = [ds1_dir, ds2_dir]
    img_id_offset = 0
    ann_id_offset = 0
    for i, (ann_file, img_dir) in enumerate(zip(ann_files, img_dirs)):
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not merged['categories']:
            merged['categories'] = data['categories']
        # Копируем изображения и меняем file_name
        id_map = {}
        for img in data['images']:
            new_id = img['id'] + img_id_offset
            id_map[img['id']] = new_id
            new_file_name = f"{Path(img_dir).name}_{img['file_name']}"
            shutil.copy2(os.path.join(img_dir, img['file_name']), os.path.join(out_images_dir, new_file_name))
            img_new = img.copy()
            img_new['id'] = new_id
            img_new['file_name'] = new_file_name
            merged['images'].append(img_new)
        for ann in data['annotations']:
            ann_new = ann.copy()
            ann_new['id'] = ann['id'] + ann_id_offset
            ann_new['image_id'] = id_map[ann['image_id']]
            merged['annotations'].append(ann_new)
        img_id_offset += max([img['id'] for img in data['images']])
        ann_id_offset += max([ann['id'] for ann in data['annotations']])
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logging.info(f"Объединено {len(merged['images'])} изображений и {len(merged['annotations'])} аннотаций. JSON: {out_json_path}")
    return out_json_path, out_images_dir

def prepare_dataset_structure_two(ds1_dir, ds2_dir, output_dir):
    """
    Готовит структуру директорий для объединённых датасетов (train/val)
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    # Копируем все изображения из обоих датасетов в обе папки (train и val)
    for src_dir in [ds1_dir, ds2_dir]:
        for tif_file in Path(src_dir).glob("*.tif"):
            shutil.copy2(tif_file, train_dir)
            shutil.copy2(tif_file, val_dir)
    return train_dir, val_dir

def register_datasets(train_json, val_json, train_images_dir, val_images_dir):
    """
    Регистрирует тренировочный и валидационный датасеты в Detectron2
    """
    register_coco_instances("train_dataset_280625", {}, train_json, train_images_dir)
    register_coco_instances("val_dataset_280625", {}, val_json, val_images_dir)
    
    # Получаем количество классов из JSON файла
    with open(train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        num_classes = len(data['categories'])
    
    return num_classes

def finetune_model(train_json, val_json, train_images_dir, val_images_dir, 
                  pretrained_weights, output_dir, num_iterations=3000):
    """
    Дообучает предварительно обученную модель на объединённом датасете
    """
    # Регистрируем датасеты
    num_classes = register_datasets(train_json, val_json, train_images_dir, val_images_dir)
    logging.info(f"Зарегистрировано {num_classes} классов")
    
    # Настраиваем конфигурацию
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    
    # Названия наборов данных
    cfg.DATASETS.TRAIN = ("train_dataset_280625",)
    cfg.DATASETS.TEST = ("val_dataset_280625",)
    
    # Директория для сохранения результатов
    cfg.OUTPUT_DIR = output_dir
    
    # Параметры загрузчика данных
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    
    # Параметры модели
    cfg.MODEL.WEIGHTS = pretrained_weights
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Параметры обучения (SOLVER)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0003  # Немного уменьшил LR для стабильности
    cfg.SOLVER.MAX_ITER = num_iterations
    cfg.SOLVER.STEPS = [int(num_iterations * 0.6), int(num_iterations * 0.8)]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.CHECKPOINT_PERIOD = 200  # Сохраняем чекпоинты чаще
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    
    # Параметры ввода (INPUT)
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 1024)  # Несколько размеров для аугментации
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.FORMAT = "BGR"
    
    # Настройка оценки
    cfg.TEST.EVAL_PERIOD = 200  # Оцениваем чаще
    
    cfg.freeze()
    
    # Создаем тренер с аугментациями
    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return cfg

def main():
    # Используем только два последних датасета
    ds1 = "dataset_180625"
    ds2 = "dataset_280625"
    temp_dataset_dir = "temp_dataset_2last"
    os.makedirs(temp_dataset_dir, exist_ok=True)
    train_json_path = os.path.join(temp_dataset_dir, "train_annotations.json")
    val_json_path = os.path.join(temp_dataset_dir, "val_annotations.json")
    train_images_dir = os.path.join(temp_dataset_dir, "train")
    val_images_dir = os.path.join(temp_dataset_dir, "val")

    # Объединяем аннотации и изображения для train и val
    logging.info("Объединяем два последних датасета для обучения и валидации...")
    merge_two_datasets(ds1, ds2, train_json_path, train_images_dir)
    merge_two_datasets(ds1, ds2, val_json_path, val_images_dir)

    # Проверяем веса
    pretrained_weights = "output/finetuned_model_r101_180625/model_0004699.pth"
    if not os.path.exists(pretrained_weights):
        logging.warning(f"Предварительно обученные веса не найдены: {pretrained_weights}")
        logging.info("Используем веса COCO по умолчанию")
        pretrained_weights = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
    output_dir = "output/finetuned_model_r101_2last"
    os.makedirs(output_dir, exist_ok=True)

    # Статистика
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    logging.info(f"Train: {len(train_data['images'])} изображений, {len(train_data['annotations'])} аннотаций")
    logging.info(f"Val: {len(val_data['images'])} изображений, {len(val_data['annotations'])} аннотаций")

    # Запуск дообучения
    cfg = finetune_model(
        train_json=train_json_path,
        val_json=val_json_path,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        pretrained_weights=pretrained_weights,
        output_dir=output_dir,
        num_iterations=3000
    )
    logging.info(f"Модель успешно дообучена и сохранена в {output_dir}")
    # Сохраняем инфо
    dataset_info = {
        "train_datasets": [ds1, ds2],
        "val_datasets": [ds1, ds2],
        "train_images": len(train_data['images']),
        "train_annotations": len(train_data['annotations']),
        "val_images": len(val_data['images']),
        "val_annotations": len(val_data['annotations']),
        "num_classes": len(train_data['categories']),
        "pretrained_weights": pretrained_weights,
        "iterations": 3000,
        "strategy": "Train/val on merged two latest datasets"
    }
    with open(os.path.join(output_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    logging.info("Информация о датасете сохранена в dataset_info.json")
    # Очистка
    logging.info("Очищаем временные файлы...")
    shutil.rmtree(temp_dataset_dir)
    logging.info("Дообучение завершено успешно!")

if __name__ == "__main__":
    main() 