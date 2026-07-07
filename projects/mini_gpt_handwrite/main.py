from mini_gpt_train import train, train_ddp, evaluate, predict

if __name__ == '__main__':
    # 单卡训练
    # train()

    # DDP 多卡训练（用 torchrun 启动）：
    #   torchrun --nproc_per_node=2 main.py --ddp
    #   torchrun --nproc_per_node=4 main.py --ddp
    import sys
    if '--ddp' in sys.argv:
        train_ddp()
    else:
        # evaluate()
        predict()
