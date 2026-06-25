import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from image_base_consumer import ImageBaseConsumer


class XceptionConsumer(ImageBaseConsumer):
    def setreqvram(self):
        self.req_vram = 7480

    def preprocess(self, file_path):
        img = Image.open(file_path).convert('RGB')
        img = img.resize((256, 256))

        img_data = np.array(img).astype(np.float32)
        img_data = (img_data / 255.0 - 0.5) / 0.5
        img_data = np.transpose(img_data, (2, 0, 1))

        return np.expand_dims(img_data, axis=0)

    def postprocess(self, raw_output):
        logits = raw_output[0]
        probs = self.softmax(logits)

        pred_idx = np.argmax(probs)
        label = "FAKE" if pred_idx == 1 else "REAL"

        return {
            "label": label,
            "logits": logits.tolist(),
            "probabilities": probs.tolist()
        }


if __name__ == "__main__":
    consumer = XceptionConsumer(model_name="xception")
    consumer.setreqvram()
    consumer.start()