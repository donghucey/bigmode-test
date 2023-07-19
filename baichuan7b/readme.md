
单机运行
python -m torch.distributed.run --nproc_per_node=8 \
    --master_port=7875 train_test.py \
--deepspeed_config deepspeed_test.json \
--gradient_checkpointing \
 &> training.log

 多机运行
需要配置hostfile免密登录

deepspeed --hostfile=hostfile --master_port=7875  train_test.py  \
--deepspeed --deepspeed_config deepspeed_test.json \
--gradient_checkpointing \
&> multi_node_training.log
