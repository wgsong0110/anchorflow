"""mmcv.ops 의 QueryAndGroup / grouping_operation 을 순수 torch 로 대체한다.

GausSim 은 mmcv 1.x 의 CUDA 확장(`mmcv._ext`)을 요구하는데, torch 2.5 / py3.11
조합의 prebuilt wheel 이 없고 인스턴스에서 컴파일하지 않는 것이 이 작업의 규칙이다.
그래서 실제로 쓰이는 두 연산만 같은 시그니처·같은 수식으로 다시 쓴다.

베이스라인 코드를 바꾸는 것이 아니라 **같은 값을 내는 구현으로 갈아끼우는 것**이다.
느릴 뿐 결과는 같아야 한다. 두 가지만 짚어둔다:

- 이 저장소의 config 는 `max_radius=None` 이라 mmcv 도 ball query 가 아니라
  **kNN** 경로를 탄다. kNN 은 cdist + topk 로 정확히 같은 이웃을 준다.
- ball query 경로도 mmcv 의 규약(반경 안에서 앞에서부터 sample_num 개, 모자라면
  첫 번째 것으로 채움)을 그대로 따라 구현해 두었다. 현재 설정에서는 안 탄다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def grouping_operation(features: torch.Tensor,
                       indices: torch.Tensor) -> torch.Tensor:
    """features (B, C, N) 에서 indices (B, npoint, nsample) 로 모은다 -> (B, C, npoint, nsample)."""
    B, C, N = features.shape
    _, npoint, nsample = indices.shape
    idx = indices.long().clamp(0, N - 1).reshape(B, 1, npoint * nsample)
    idx = idx.expand(B, C, npoint * nsample)
    return torch.gather(features, 2, idx).reshape(B, C, npoint, nsample)


def knn(k: int, xyz: torch.Tensor, center_xyz: torch.Tensor,
        transposed: bool = False) -> torch.Tensor:
    """center_xyz 각 점에 대한 xyz 안의 최근접 k 개. 반환 (B, k, npoint) -- mmcv 와 같은 축 순서."""
    if transposed:
        xyz = xyz.transpose(1, 2).contiguous()
        center_xyz = center_xyz.transpose(1, 2).contiguous()
    d = torch.cdist(center_xyz.float(), xyz.float())          # (B, npoint, N)
    idx = d.topk(min(k, d.shape[-1]), dim=-1, largest=False).indices   # (B, npoint, k)
    return idx.transpose(1, 2).int().contiguous()


def ball_query(min_radius: float, max_radius: float, sample_num: int,
               xyz: torch.Tensor, center_xyz: torch.Tensor) -> torch.Tensor:
    """(B, npoint, sample_num). mmcv 규약: 껍질 [min, max) 안의 것만, 모자라면 첫 번째로 채운다."""
    d = torch.cdist(center_xyz.float(), xyz.float())           # (B, npoint, N)
    ok = (d < max_radius) & (d >= min_radius)
    # 반경 밖은 뒤로 밀어내고 거리순으로 정렬 -> 앞에서 sample_num 개를 집는다
    big = d.max().detach() + 1.0
    order = torch.where(ok, d, d + big).argsort(dim=-1)
    idx = order[..., :sample_num]
    valid = torch.gather(ok, 2, idx)
    first = idx[..., :1].expand_as(idx)
    idx = torch.where(valid, idx, first)
    # 하나도 없으면 mmcv 는 0 을 채운다
    none_ok = ~ok.any(dim=-1, keepdim=True)
    idx = torch.where(none_ok.expand_as(idx), torch.zeros_like(idx), idx)
    return idx.int().contiguous()


class QueryAndGroup(nn.Module):
    """mmcv.ops.QueryAndGroup 과 같은 인자·같은 반환."""

    def __init__(self, max_radius, sample_num, min_radius=0, use_xyz=True,
                 return_grouped_xyz=False, normalize_xyz=False,
                 uniform_sample=False, return_unique_cnt=False,
                 return_grouped_idx=False):
        super().__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.sample_num = sample_num
        self.use_xyz = use_xyz
        self.return_grouped_xyz = return_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.uniform_sample = uniform_sample
        self.return_unique_cnt = return_unique_cnt
        self.return_grouped_idx = return_grouped_idx
        if self.max_radius is None:
            assert not self.normalize_xyz and not self.return_unique_cnt, \
                "kNN 경로에서는 normalize_xyz / return_unique_cnt 를 쓸 수 없다 (mmcv 와 동일)"

    def forward(self, points_xyz, center_xyz, features=None):
        if self.max_radius is None:
            idx = knn(self.sample_num, points_xyz, center_xyz, False)
            idx = idx.transpose(1, 2).contiguous()             # (B, npoint, k)
        else:
            idx = ball_query(self.min_radius, self.max_radius, self.sample_num,
                             points_xyz, center_xyz)

        unique_cnt = None
        if self.return_unique_cnt:
            unique_cnt = torch.stack([
                torch.stack([idx[b, i].unique().numel()
                             for i in range(idx.shape[1])])
                for b in range(idx.shape[0])]).to(idx.device)

        xyz_trans = points_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)       # (B, 3, npoint, k)
        grouped_xyz_diff = grouped_xyz - center_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz_diff = grouped_xyz_diff / self.max_radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz_diff, grouped_features],
                                         dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "features 가 없으면 use_xyz 여야 한다"
            new_features = grouped_xyz_diff

        ret = [new_features]
        if self.return_grouped_xyz:
            ret.append(grouped_xyz)
        if self.return_unique_cnt:
            ret.append(unique_cnt)
        if self.return_grouped_idx:
            ret.append(idx)
        return ret[0] if len(ret) == 1 else tuple(ret)
