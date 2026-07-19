"""
Xception (Chollet, 2017) — depthwise-separable conv 기반 딥페이크 탐지 분류기.

- timm 미사용, 사전학습 없음 (무작위 초기화)
- AdaptiveAvgPool로 임의 입력 크기(256x256) 지원
- 출력: [B, num_classes] 로짓
- clean-room 재작성 (논문 구조만 따름, 가중치/코드 이식 없음)
- 상업적 사용 안전
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """Depthwise(그룹=in_ch) 3x3 + Pointwise 1x1. bias 없음(뒤에 BN)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding,
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class Block(nn.Module):
    """SeparableConv 반복 + 1x1 skip 잔차. stride!=1 이면 MaxPool로 다운샘플."""

    def __init__(self, in_ch: int, out_ch: int, reps: int,
                 stride: int = 1, start_with_relu: bool = True,
                 grow_first: bool = True):
        super().__init__()

        # skip connection
        self.skip = None
        self.skip_bn = None
        if out_ch != in_ch or stride != 1:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_ch)

        rep: list[nn.Module] = []
        filters = in_ch

        if grow_first:
            rep += [
                nn.ReLU(inplace=False),
                SeparableConv2d(in_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
            ]
            filters = out_ch

        for _ in range(reps - 1):
            rep += [
                nn.ReLU(inplace=False),
                SeparableConv2d(filters, filters, 3, 1, 1),
                nn.BatchNorm2d(filters),
            ]

        if not grow_first:
            rep += [
                nn.ReLU(inplace=False),
                SeparableConv2d(in_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
            ]

        if not start_with_relu:
            rep = rep[1:]  # 엔트리 첫 블록: 첫 ReLU 제거

        if stride != 1:
            rep += [nn.MaxPool2d(3, stride, 1)]

        self.rep = nn.Sequential(*rep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rep(x)
        skip = x if self.skip is None else self.skip_bn(self.skip(x))
        return out + skip


class Xception(nn.Module):
    """
    Entry Flow → Middle Flow(x8) → Exit Flow → [B, num_classes] 로짓.

    Args:
        num_classes: 분류 클래스 수 (기본 2: real/fake)
        in_chans: 입력 채널 수 (기본 3: RGB)
    """

    def __init__(self, num_classes: int = 2, in_chans: int = 3):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

        # 스템
        self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Entry Flow
        self.block1 = Block(64, 128, 2, stride=2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, stride=2, start_with_relu=True, grow_first=True)

        # Middle Flow x8 (728채널, 항등 잔차, 다운샘플 없음)
        self.middle = nn.Sequential(
            *[Block(728, 728, 3, stride=1) for _ in range(8)]
        )

        # Exit Flow
        self.block12 = Block(728, 1024, 2, stride=2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # 분류 헤드
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 스템
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Entry Flow
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle Flow
        x = self.middle(x)

        # Exit Flow
        x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        # 분류
        x = self.pool(x).flatten(1)
        return self.fc(x)