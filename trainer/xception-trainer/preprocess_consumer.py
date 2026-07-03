# trainer/xception-trainer/preprocess_consumer.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))

import numpy as np
from PIL import Image
from base_preprocess_consumer import BasePreprocessConsumer


class XceptionPreprocessConsumer(BasePreprocessConsumer):

    def preprocess(self, image_path: str) -> bytes:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((256, 256))
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 255.0 - 0.5) / 0.5  # [-1, 1] 정규화
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
        return img_array.tobytes()


if __name__ == "__main__":
    consumer = XceptionPreprocessConsumer()
    consumer.run()