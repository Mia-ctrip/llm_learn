#!/usr/bin/env python3
"""
驱动安装/升级验收脚本：按 Layer 3 -> 6 逐层验证，不只看nvidia-smi。
用法: python verify_driver_upgrade.py
依赖: 已安装驱动 + torch (CUDA版本)
"""

import os
import subprocess
import sys


def check_layer3_nvidia_smi():
    """Layer 3: NVML/管理面层"""
    print("=" * 60)
    print("[Layer 3] nvidia-smi 管理面检查")
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,driver_version,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        print(result.stdout.strip())
        print("[PASS] nvidia-smi 正常响应")
        return True
    except Exception as e:
        print(f"[FAIL] nvidia-smi 执行失败: {e}")
        return False


def check_layer4_cuda_context():
    """Layer 4: CUDA Driver层，能否建立Context"""
    print("=" * 60)
    print("[Layer 4] CUDA Context 建立检查")
    try:
        import torch
    except ImportError:
        print("[FAIL] 未安装torch，无法继续后续检查")
        return False

    try:
        if not torch.cuda.is_available():
            print("[FAIL] torch.cuda.is_available() 返回 False")
            return False
        device_count = torch.cuda.device_count()
        print(f"[PASS] 检测到 {device_count} 张可用GPU")
        for i in range(device_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        return device_count > 0
    except Exception as e:
        print(f"[FAIL] CUDA Context建立异常: {e}")
        return False


def check_layer5_compute():
    """Layer 5: CUDA Runtime层，真实分配显存并跑一次计算"""
    print("=" * 60)
    print("[Layer 5] 显存分配 + 计算检查")
    import torch
    try:
        device = torch.device("cuda:0")
        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)
        c = a @ b
        torch.cuda.synchronize()
        checksum = c.sum().item()
        mem_mb = torch.cuda.memory_allocated(device) / 1024**2
        print(f"[PASS] 矩阵乘法完成，结果校验值: {checksum:.2f}，显存占用: {mem_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"[FAIL] GPU计算异常: {e}")
        return False


def check_layer6_multi_gpu_nccl():
    """Layer 6: 应用/框架层，多卡场景下NCCL通信是否正常"""
    print("=" * 60)
    print("[Layer 6] 多卡NCCL通信检查")
    import torch
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("[SKIP] 只有1张GPU，跳过多卡通信检查")
        return True
    if os.name == "nt":
        print("[SKIP] NCCL仅支持Linux，Windows环境跳过（可改用gloo后端自行验证）")
        return True

    try:
        import torch.distributed as dist
        import torch.multiprocessing as mp

        def worker(rank, world_size):
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29501"
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            tensor = torch.ones(1, device=f"cuda:{rank}") * (rank + 1)
            dist.all_reduce(tensor)
            expected = sum(range(1, world_size + 1))
            dist.destroy_process_group()
            if abs(tensor.item() - expected) > 1e-3:
                raise RuntimeError(f"rank{rank} all_reduce结果异常: {tensor.item()} != {expected}")

        mp.spawn(worker, args=(device_count,), nprocs=device_count, join=True)
        print(f"[PASS] {device_count}卡 NCCL all_reduce 通信正常")
        return True
    except Exception as e:
        print(f"[FAIL] 多卡NCCL通信异常: {e}")
        return False


def main():
    results = {}
    results["Layer3 nvidia-smi"] = check_layer3_nvidia_smi()
    results["Layer4 CUDA Context"] = check_layer4_cuda_context()
    if results["Layer4 CUDA Context"]:
        results["Layer5 显存计算"] = check_layer5_compute()
        results["Layer6 多卡NCCL"] = check_layer6_multi_gpu_nccl()

    print("=" * 60)
    print("验收结果汇总:")
    all_pass = True
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_pass = all_pass and ok

    print("=" * 60)
    if all_pass:
        print("驱动验收通过：Layer 3~6 均正常，业务可用")
        sys.exit(0)
    else:
        print("驱动验收失败：请结合 dmesg -T | grep xid 及 nvidia-smi -q 进一步定位")
        sys.exit(1)


if __name__ == "__main__":
    main()
