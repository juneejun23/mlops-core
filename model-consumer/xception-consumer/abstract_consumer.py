from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from aiokafka import AIOKafkaProducer


class AbstractConsumer(ABC):
    @abstractmethod
    def setreqvram(self) -> None:
        """이 모델에 필요한 VRAM(MiB)을 self.req_vram에 설정"""
        pass

    @abstractmethod
    def preprocess(self, file_path: str) -> np.ndarray:
        """로컬 파일 경로를 받아 모델 입력 형태의 numpy array로 변환"""
        pass

    @abstractmethod
    def postprocess(self, raw_output: np.ndarray) -> dict:
        """Triton 추론 결과를 받아 구조화된 결과 dict로 변환"""
        pass

    @abstractmethod
    def _download_s3_file(self, input_key: str) -> str:
        """MinIO에서 파일을 임시 경로로 다운로드"""
        pass

    @abstractmethod
    async def _is_cluster_saturated(self) -> bool:
        """등록된 Triton 서버가 하나도 없는지 확인"""
        pass

    @abstractmethod
    def _calculate_cost(self, server_stats: dict) -> int:
        """서버 상태를 보고 스케줄링 비용 계산"""
        pass

    @abstractmethod
    async def _schedule_server(self) -> str:
        """사용 가능한 Triton 서버를 찾아 IP를 반환"""
        pass

    @abstractmethod
    async def _fetch_and_sync_vram(self, server_ip: str, server_key: str) -> Optional[int]:
        """Triton 메트릭에서 VRAM 사용량을 가져와 Redis에 동기화"""
        pass

    @abstractmethod
    async def _try_lock_and_load(self, target_server: dict) -> bool:
        """서버 락 획득 후 모델 로드"""
        pass

    @abstractmethod
    async def _rollback_state(self, target_server: dict) -> None:
        """로드 실패 시 서버 상태를 ACTIVE로 복구"""
        pass

    @abstractmethod
    async def _process_message(self, msg_value: dict, producer: AIOKafkaProducer) -> None:
        """Kafka 메시지 하나에 대한 전체 추론 파이프라인 실행"""
        pass

    @abstractmethod
    async def run(self) -> None:
        """메인 비동기 루프: Kafka 연결, 메시지 polling, 처리"""
        pass

    @abstractmethod
    def start(self) -> None:
        """비동기 루프를 실행하는 동기 진입점"""
        pass