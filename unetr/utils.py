import base64
from django.conf import settings
import os

def decodeImage(imgstring, fileName):
    filename_path = os.path.join(settings.BASE_DIR, "unetr", "unetr_model_output", "decode", fileName)
    imgdata = base64.b64decode(imgstring)
    with open(filename_path, "wb") as f:
        f.write(imgdata)
        f.close()
    print("decode done", "="*100)


def encodeImageIntoBase64(croppedImagePath):
    filename_path = os.path.join(settings.BASE_DIR, "unetr", "unetr_model_output", "predict", croppedImagePath)
    with open(filename_path, "rb") as f:
        print("encode done", "="*100)
        return base64.b64encode(f.read())
    