
依赖环境参考
https://github.com/baichuan-inc/Baichuan-7B


单机运行
python -m torch.distributed.run --nproc_per_node=8 \
    --master_port=7875 train_test.py \
--deepspeed_config deepspeed_test.json \
--gradient_checkpointing \
 &> training.log

 多机运行
需要配置hostfile免密登录

NCCL_SOCKET_IFNAME=enp141s0f0 NCCL_IB_DISABLE=0  NCCL_IB_GID_INDEX=3 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 NCCL_DEBUG=INFO \
deepspeed --hostfile=hostfile --master_port=7875  train_test.py  \
--deepspeed --deepspeed_config deepspeed_test.json \
--gradient_checkpointing \
&> multi_node_training.log
