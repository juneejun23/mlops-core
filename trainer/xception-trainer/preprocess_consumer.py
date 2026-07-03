import numpy as np
from PIL import Image
from base_preprocess_consumer import BasePreprocessConsumer


class XceptionPreprocessConsumer(BasePreprocessConsumer):

    def __init__(self):
        super().__init__()
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            from mtcnn import MTCNN
            self._detector = MTCNN()
        return self._detector

    def preprocess(self, image_path: str, face_based: bool = False):
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        if face_based:
            detector = self._get_detector()
            faces = detector.detect_faces(img_array)
            if len(faces) == 0:
                return None  # 얼굴 미검출 → 스킵
            # 가장 큰 얼굴 선택
            face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            img_array = img_array[y:y+h, x:x+w]
            img = Image.fromarray(img_array)

        img = img.resize((256, 256))
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 255.0 - 0.5) / 0.5
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array.tobytes()


if __name__ == "__main__":
    consumer = XceptionPreprocessConsumer()
    consumer.run()