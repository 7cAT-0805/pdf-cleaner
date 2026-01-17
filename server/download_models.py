import easyocr
import os

def download():
    # 獲取當前腳本所在的目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models')
    
    print(f"Pre-downloading EasyOCR models to {model_path}...")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # 使用 model_storage_directory 指定路徑
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=False, model_storage_directory=model_path)
    print(f"Models downloaded successfully to {model_path}.")

if __name__ == "__main__":
    download()
