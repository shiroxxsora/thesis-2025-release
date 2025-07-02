import os
import json
import random
import numpy as np
import cv2 # OpenCV может понадобиться для некоторых операций загрузки/аугментации Detectron2
import torch
import copy
import logging # Используем logging для лучшего вывода

# Detectron2 imports
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util # Необходим для COCOEvaluator

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Пользовательский Оценщик (F2-логика закомментирована) ---
class F2Evaluator(COCOEvaluator):
    def __init__(self, dataset_name, output_dir=None):
        super().__init__(dataset_name, output_dir=output_dir)
        # self._f2_scores = [] # F2-логика временно отключена

    def _evaluate_predictions_on_coco(self, coco_gt, coco_results, iou_type="segm", **kwargs):
        """
        Оценивает предсказания с использованием стандартных метрик COCO.
        ПРИМЕЧАНИЕ: Логика F2-score закомментирована из-за некорректности для instance segmentation в текущем виде.
        """
        logging.info("Running standard COCO evaluation...")
        results = super()._evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, **kwargs)

        # ----------------------------------------------------------------------
        # --- Начало закомментированной F2-логики ---
        # ПРЕДУПРЕЖДЕНИЕ: Следующий код НЕКОРРЕКТНО вычисляет F2-score для instance segmentation,
        # так как он не выполняет сопоставление предсказанных экземпляров с ground truth экземплярами.
        # Он оставлен здесь только для демонстрации первоначальной идеи, но не должен использоваться.
        # ----------------------------------------------------------------------
        # if iou_type == "segm" and len(coco_results) > 0:
        #     f2_scores_per_image = [] # Неправильный способ агрегации
        #     num_valid_f2 = 0
        #     for result in coco_results:
        #         if "segmentation" not in result:
        #             continue

        #         pred_mask_rle = result["segmentation"]
        #         image_id = result["image_id"]
        #         # Получаем ВСЕ аннотации для этого изображения
        #         gt_anns_ids = coco_gt.getAnnIds(imgIds=[image_id])
        #         gt_anns = coco_gt.loadAnns(gt_anns_ids)

        #         if not gt_anns: # Нет ground truth для этого изображения
        #             continue

        #         # --- НЕКОРРЕКТНАЯ ЛОГИКА ---
        #         # Сравнение первого предсказания с первой GT маской - неверно!
        #         try:
        #             pred_mask = mask_util.decode(pred_mask_rle)
        #             gt_mask = coco_gt.annToMask(gt_anns[0]) # Берем маску только первой GT аннотации

        #             # Вычисляем TP, FP, FN на уровне всего изображения (не экземпляра)
        #             tp = np.sum(np.logical_and(pred_mask, gt_mask))
        #             fp = np.sum(np.logical_and(pred_mask, np.logical_not(gt_mask)))
        #             fn = np.sum(np.logical_and(np.logical_not(pred_mask), gt_mask))

        #             # Вычисляем precision и recall
        #             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        #             recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        #             # Вычисляем F2-score
        #             beta = 2
        #             f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
        #             f2_scores_per_image.append(f2)
        #             num_valid_f2 +=1
        #         except Exception as e:
        #             logging.warning(f"Could not calculate F2 for an instance on image {image_id}: {e}")
        #             continue
        #         # --- КОНЕЦ НЕКОРРЕКТНОЙ ЛОГИКИ ---

        #     if num_valid_f2 > 0:
        #         avg_f2 = np.mean(f2_scores_per_image) # Усреднение некорректных F2
        #         # Не добавляем в стандартные результаты, чтобы не вводить в заблуждение
        #         logging.info(f"[INFO] Incorrectly calculated average F2-score: {avg_f2:.4f} (Based on {num_valid_f2} calculations)")
        #         # self._f2_scores.append(avg_f2) # Не сохраняем некорректный результат
        #     else:
        #          logging.info("[INFO] Could not calculate any F2 scores with the current (incorrect) logic.")
        # ----------------------------------------------------------------------
        # --- Конец закомментированной F2-логики ---
        # ----------------------------------------------------------------------

        return results

# --- Пользовательский Тренер ---
class CustomTrainer(DefaultTrainer):
    """
    Кастомный тренер с возможностью настройки аугментаций и оценщика.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Создает загрузчик данных для обучения с кастомными аугментациями.
        """
        # Здесь мы могли бы использовать кастомный DatasetMapper, если бы хотели Albumentations
        # mapper = CustomDatasetMapper(cfg, is_train=True)
        # return build_detection_train_loader(cfg, mapper=mapper)

        # Используем стандартный маппер с аугментациями Detectron2
        return build_detection_train_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Создает оценщик. Возвращаем стандартный COCOEvaluator.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        # Возвращаем стандартный оценщик COCO
        logging.info(f"Using standard COCOEvaluator for dataset '{dataset_name}'.")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
        # Если бы F2Evaluator был корректен, вернули бы его:
        # return F2Evaluator(dataset_name, output_dir=output_folder)

    # Метод build_train_augmentation больше не нужен, т.к. аугментации
    # настраиваются через cfg.INPUT и используются стандартным DatasetMapper'ом.
    # Если нужны кастомные аугментации, не входящие в стандартные T.*,
    # то нужно создавать свой DatasetMapper.
    # @classmethod
    # def build_train_augmentation(cls, cfg):
    #     """
    #     (Устарело, если используем стандартный маппер)
    #     Создает пайплайн аугментации для обучения
    #     """
    #     augmentation = [
    #         T.ResizeShortestEdge(
    #             cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, sample_style="choice" # Используем sample_style
    #         ),
    #         # Добавляем стандартные аугментации Detectron2
    #         T.RandomBrightness(0.8, 1.2),
    #         T.RandomContrast(0.8, 1.2),
    #         T.RandomSaturation(0.8, 1.2),
    #         # T.RandomRotation([-15, 15]), # RandomRotation не является стандартным в detectron2.data.transforms
    #         T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    #         # T.RandomCrop("relative_range", (0.8, 0.8)), # Используем настройки cfg.INPUT.CROP
    #         # T.RandomLighting(0.7), # RandomLighting не является стандартным
    #     ]
    #     return augmentation


# --- Настройка Конфигурации ---
def setup_cfg(train_dataset_name, val_dataset_name, num_classes, output_dir):
    """
    Создает и настраивает конфигурацию Detectron2.
    """
    cfg = get_cfg()
    # Загружаем базовую конфигурацию Mask R-CNN R-101 FPN
    #cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Названия наборов данных
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)

    # Директория для сохранения результатов
    cfg.OUTPUT_DIR = output_dir

    # Параметры загрузчика данных
    cfg.DATALOADER.NUM_WORKERS = 2 # Увеличьте, если у вас много CPU

    # Параметры модели
    # Загружаем веса, предобученные на COCO
    #cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # Количество RoIs для обработки на каждом изображении
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Количество ваших классов (только передний план)

    # Параметры обучения (SOLVER)
    cfg.SOLVER.IMS_PER_BATCH = 2 # Количество изображений на батч (GPU). Уменьшите, если не хватает VRAM
    cfg.SOLVER.BASE_LR = 0.0005 # Базовая скорость обучения
    cfg.SOLVER.MAX_ITER = 7500    # Максимальное количество итераций обучения
    cfg.SOLVER.OPTIMIZER = "ADAMW" # Оптимизатор
    cfg.SOLVER.STEPS = [3000, 4500, 5500] # Итерации, на которых LR будет уменьшен
    cfg.SOLVER.GAMMA = 0.1 # Коэффициент уменьшения LR (был 0.05, стандартно 0.1)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500 # Как часто сохранять чекпоинт

    # Параметры ввода (INPUT)
    # Включение стандартных аугментаций через конфиг
    cfg.INPUT.RANDOM_FLIP = "horizontal" # Включаем горизонтальный флип
    # Настройка кадрирования (если нужно)
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.8, 0.8]
    # Настройка изменения размера (многомасштабное обучение)
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800) # Случайный выбор мин. размера
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.FORMAT = "BGR" # Стандартный формат цвета для Detectron2

    # Настройка оценки
    cfg.TEST.EVAL_PERIOD = 500  # Оцениваем на валидационном наборе каждые N итераций

    cfg.freeze() # Замораживаем конфиг после настройки
    return cfg

# --- Разделение Датасета ---
def split_dataset(annotations_file, train_ratio=0.8, seed=42):
    """
    Разделяет датасет COCO на тренировочную и валидационную части по изображениям.
    """
    logging.info(f"Splitting dataset from {annotations_file} with train_ratio={train_ratio}")
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Annotation file not found: {annotations_file}")
        return None, None
    except json.JSONDecodeError:
         logging.error(f"Error decoding JSON from file: {annotations_file}")
         return None, None

    if "images" not in data or not data["images"]:
         logging.error("No 'images' found in the annotation file.")
         return None, None

    # Фиксируем seed для воспроизводимости разделения
    random.seed(seed)

    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)

    split_idx = int(len(image_ids) * train_ratio)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    logging.info(f"Total images: {len(image_ids)}. Training images: {len(train_ids)}, Validation images: {len(val_ids)}")

    # Создаем базовую структуру для train и val аннотаций
    base_data_structure = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data.get('categories', []) # Важно сохранить категории!
    }
    train_data = copy.deepcopy(base_data_structure)
    train_data['images'] = []
    train_data['annotations'] = []

    val_data = copy.deepcopy(base_data_structure)
    val_data['images'] = []
    val_data['annotations'] = []

    # Распределяем изображения
    img_id_map = {img['id']: img for img in data['images']}
    for img_id in train_ids:
        if img_id in img_id_map:
            train_data['images'].append(img_id_map[img_id])
    for img_id in val_ids:
         if img_id in img_id_map:
            val_data['images'].append(img_id_map[img_id])

    # Распределяем аннотации
    if "annotations" in data:
        for ann in data['annotations']:
            if ann['image_id'] in train_ids:
                train_data['annotations'].append(ann)
            elif ann['image_id'] in val_ids:
                val_data['annotations'].append(ann)
    else:
        logging.warning("No 'annotations' key found in the source file.")

    logging.info("Dataset splitting complete.")
    return train_data, val_data

# --- Основная функция ---
def main():
    # --- Конфигурация путей ---
    COMBINED_DATASET_DIR = "Combined_Dataset" # Папка с общим датасетом (где лежит папка images)
    COMBINED_ANNOTATIONS_FILE = os.path.join(COMBINED_DATASET_DIR, "annotations.json") # Имя файла общей разметки
    IMAGES_DIR = os.path.join(COMBINED_DATASET_DIR, "images") # Папка с изображениями

    TRAIN_ANNOTATIONS_DIR = "train_annotations"
    VAL_ANNOTATIONS_DIR = "val_annotations"
    TRAIN_JSON_PATH = os.path.join(TRAIN_ANNOTATIONS_DIR, "instances_train.json")
    VAL_JSON_PATH = os.path.join(VAL_ANNOTATIONS_DIR, "instances_val.json")

    OUTPUT_DIR = "./output_maskrcnn_r50" # Директория для сохранения модели и логов

    TRAIN_DATASET_NAME = "territory_train"
    VAL_DATASET_NAME = "territory_val"

    NUM_CLASSES = 1 # Количество ваших классов (например, 1 для "территория")
    TRAIN_RATIO = 0.8 # Доля данных для обучения

    # --- 1. Разделение датасета ---
    os.makedirs(TRAIN_ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(VAL_ANNOTATIONS_DIR, exist_ok=True)

    train_data, val_data = split_dataset(COMBINED_ANNOTATIONS_FILE, train_ratio=TRAIN_RATIO)

    if train_data is None or val_data is None:
        logging.error("Failed to split dataset. Exiting.")
        return

    # Сохраняем разделенные аннотации
    try:
        with open(TRAIN_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Training annotations saved to {TRAIN_JSON_PATH}")
        with open(VAL_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Validation annotations saved to {VAL_JSON_PATH}")
    except Exception as e:
        logging.error(f"Failed to save split annotation files: {e}")
        return

    # --- 2. Регистрация наборов данных ---
    # Удаляем старые регистрации, если они были
    if TRAIN_DATASET_NAME in DatasetCatalog.list():
        DatasetCatalog.remove(TRAIN_DATASET_NAME)
        MetadataCatalog.remove(TRAIN_DATASET_NAME)
    if VAL_DATASET_NAME in DatasetCatalog.list():
         DatasetCatalog.remove(VAL_DATASET_NAME)
         MetadataCatalog.remove(VAL_DATASET_NAME)

    register_coco_instances(
        name=TRAIN_DATASET_NAME,
        metadata={}, # Можно добавить метаданные, например, {'thing_classes': ['territory']}
        json_file=TRAIN_JSON_PATH,
        image_root=IMAGES_DIR # <--- ИСПРАВЛЕНО: Указываем на папку с изображениями
    )
    register_coco_instances(
        name=VAL_DATASET_NAME,
        metadata={},
        json_file=VAL_JSON_PATH,
        image_root=IMAGES_DIR # <--- ИСПРАВЛЕНО: Указываем на папку с изображениями
    )
    logging.info(f"Datasets '{TRAIN_DATASET_NAME}' and '{VAL_DATASET_NAME}' registered.")

    # (Опционально) Задаем метаданные для визуализации
    MetadataCatalog.get(TRAIN_DATASET_NAME).thing_classes = ["plot"] # Замените на имя вашего класса
    MetadataCatalog.get(VAL_DATASET_NAME).thing_classes = ["plot"]

    # --- 3. Настройка конфигурации ---
    cfg = setup_cfg(
        train_dataset_name=TRAIN_DATASET_NAME,
        val_dataset_name=VAL_DATASET_NAME,
        num_classes=NUM_CLASSES,
        output_dir=OUTPUT_DIR
    )
    logging.info(f"Configuration setup complete. Output directory: {cfg.OUTPUT_DIR}")
    # Вывод конфигурации (опционально, для отладки)
    # logging.info("Effective configuration:\n" + cfg.dump())


    # --- 4. Обучение ---
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Используем CustomTrainer, который теперь использует стандартный COCOEvaluator
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False) # resume=True для продолжения обучения с последнего чекпоинта
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")

    # --- 5. Оценка после обучения (опционально) ---
    logging.info("Starting evaluation on the validation set after training...")
    evaluator = COCOEvaluator(VAL_DATASET_NAME, output_dir=os.path.join(cfg.OUTPUT_DIR, "final_eval"))
    # Загружаем лучшую модель (обычно model_final.pth)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    val_loader = build_detection_test_loader(cfg, VAL_DATASET_NAME)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    logging.info("Final evaluation results:")
    logging.info(results)


if __name__ == "__main__":
    main()