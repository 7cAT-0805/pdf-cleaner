import easyocr
import os

def download():
    print("Pre-downloading EasyOCR models...")
    # This will download the models to the default directory (~/.EasyOCR/model)
    # Perform this during building to avoid OOM during runtime.
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)
    print("Models downloaded successfully.")

if __name__ == "__main__":
    download()
