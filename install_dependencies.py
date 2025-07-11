import subprocess
import sys
import os

def install_package(package):
    """Установка пакета через pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Установлен {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Ошибка установки {package}")
        return False

def main():
    print("Установка дополнительных зависимостей для комплексного анализа...")
    
    # Список необходимых пакетов
    packages = [
        "matplotlib",
        "reportlab",
        "geopandas",
        "shapely",
        "rasterio",
        "gdal",
        "opencv-python",
        "numpy",
        "pandas",
        "openpyxl",
        "torch",
        "torchvision"
    ]
    
    print("Проверка и установка пакетов...")
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} уже установлен")
        except ImportError:
            print(f"Устанавливаю {package}...")
            install_package(package)
    
    print("\nПроверка Detectron2...")
    try:
        import detectron2
        print("✓ Detectron2 установлен")
    except ImportError:
        print("✗ Detectron2 не найден. Установите его вручную:")
        print("pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html")
    
    print("\nУстановка завершена!")

if __name__ == "__main__":
    main() 