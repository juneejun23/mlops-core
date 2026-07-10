import numpy as np
from PIL import Image
from base_preprocess_consumer import BasePreprocessConsumer
import io

class XceptionPreprocessConsumer(BasePreprocessConsumer):

    def __init__(self):
        super().__init__()
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            import torch
            from facenet_pytorch import MTCNN
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._detector = MTCNN(keep_all=True, device=device)
        return self._detector

    def preprocess(self, image_path: str, face_based: bool = False):
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        if face_based:
            detector = self._get_detector()
            boxes, _ = detector.detect(img_array)  # boxes: [[x1,y1,x2,y2], ...]
            if boxes is None or len(boxes) == 0:
                return None
            # 가장 큰 얼굴 선택 (x2-x1) * (y2-y1) 기준
            box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            img_array = img_array[y1:y2, x1:x2]
            img = Image.fromarray(img_array)

        img = img.resize((256, 256))
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 255.0 - 0.5) / 0.5
        img_array = np.transpose(img_array, (2, 0, 1))
        buf = io.BytesIO()
        np.save(buf, img_array)
        return buf.getvalue()


if __name__ == "__main__":
    consumer = XceptionPreprocessConsumer()
    consumer.run()# trigger rebuild
