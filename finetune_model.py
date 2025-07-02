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
from detectron2.data.detection_utils import annotations_to_instances
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
        T.RandomLighting(0.7),
        T.RandomRotation([-15, 15]),
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

def prepare_dataset_structure(source_dir, output_dir):
    """
    Подготавливает структуру директорий для датасета
    """
    # Создаем основные директории
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Копируем GeoTIFF файлы
    for tif_file in Path(source_dir).glob("**/*.tif"):
        shutil.copy2(tif_file, train_dir)
        shutil.copy2(tif_file, val_dir)
    
    return train_dir, val_dir

def register_datasets(train_json, val_json, train_images_dir, val_images_dir):
    """
    Регистрирует тренировочный и валидационный датасеты в Detectron2
    """
    register_coco_instances("train_dataset", {}, train_json, train_images_dir)
    register_coco_instances("val_dataset", {}, val_json, val_images_dir)
    
    # Получаем количество классов из JSON файла
    with open(train_json, 'r', encoding='cp1251') as f:
        data = json.load(f)
        num_classes = len(data['categories'])
    
    return num_classes

def finetune_model(train_json, val_json, train_images_dir, val_images_dir, 
                  pretrained_weights, output_dir, num_iterations=1000):
    """
    Дообучает предварительно обученную модель на новом датасете
    """
    # Регистрируем датасеты
    num_classes = register_datasets(train_json, val_json, train_images_dir, val_images_dir)
    
    # Настраиваем конфигурацию
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    
    # Названия наборов данных
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)
    
    # Директория для сохранения результатов
    cfg.OUTPUT_DIR = output_dir
    
    # Параметры загрузчика данных
    cfg.DATALOADER.NUM_WORKERS = 2  # Уменьшаем количество воркеров
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    
    # Параметры модели
    cfg.MODEL.WEIGHTS = pretrained_weights
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # Уменьшаем размер батча на изображение
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Параметры обучения (SOLVER)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = num_iterations
    cfg.SOLVER.STEPS = [int(num_iterations * 0.6), int(num_iterations * 0.8)]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    
    # Параметры ввода (INPUT)
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)  # Уменьшаем количество вариантов размеров
    cfg.INPUT.MAX_SIZE_TRAIN = 1024  # Уменьшаем максимальный размер
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.FORMAT = "BGR"
    
    # Настройка оценки
    cfg.TEST.EVAL_PERIOD = 100
    
    cfg.freeze()
    
    # Создаем тренер с аугментациями
    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return cfg

def main():
    # Пути к данным
    processed_dir = "dataset_180625"
    source_annotations = os.path.join(processed_dir, "annotations", "instances_default.json")
    
    # Создаем временную директорию для разделенного датасета
    temp_dataset_dir = "temp_dataset_180625"
    os.makedirs(temp_dataset_dir, exist_ok=True)
    
    # Разделяем датасет
    train_json, val_json = split_dataset(source_annotations, train_ratio=0.8)
    
    if train_json is None or val_json is None:
        logging.error("Не удалось разделить датасет")
        return
    
    # Сохраняем разделенные аннотации
    train_json_path = os.path.join(temp_dataset_dir, "train_annotations.json")
    val_json_path = os.path.join(temp_dataset_dir, "val_annotations.json")
    
    with open(train_json_path, 'w', encoding='cp1251') as f:
        json.dump(train_json, f)
    with open(val_json_path, 'w', encoding='cp1251') as f:
        json.dump(val_json, f)
    
    # Подготавливаем структуру директорий
    train_images_dir, val_images_dir = prepare_dataset_structure(processed_dir, temp_dataset_dir)
    
    # Путь к предварительно обученным весам
    pretrained_weights = "output/finetuned_model_r101_280625/model_final.pth"
    
    # Директория для сохранения результатов
    output_dir = "output/finetuned_model_r101_290625"
    
    # Создаем директорию для выходных данных
    os.makedirs(output_dir, exist_ok=True)
    
    # Запускаем дообучение
    cfg = finetune_model(
        train_json=train_json_path,
        val_json=val_json_path,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        pretrained_weights=pretrained_weights,
        output_dir=output_dir,
        num_iterations=5000
    )
    
    logging.info(f"Модель успешно дообучена и сохранена в {output_dir}")
    
    # Очищаем временные файлы
    shutil.rmtree(temp_dataset_dir)

if __name__ == "__main__":
    main() 