# trainer/xception-trainer/train_consumer.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))

import logging
from base_train_consumer import BaseTrainConsumer


class XceptionTrainConsumer(BaseTrainConsumer):

    def train(self, payload: dict):
        training_job_id = payload["training_job_id"]
        real_keys = payload["real_keys"]
        fake_keys = payload["fake_keys"]
        logging.info(f"[STUB] Training {training_job_id}: real={len(real_keys)}, fake={len(fake_keys)}")
        # TODO: 실제 학습 로직 추가


if __name__ == "__main__":
    consumer = XceptionTrainConsumer()
    consumer.run()