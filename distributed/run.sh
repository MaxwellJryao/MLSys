# ddp
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="12355"

# python distributed_data_parallel.py

# torchrun ddp
# torchrun --standalone --nproc_per_node=2 ddp_torchrun.py 10 10

# fsdp
python fsdp.py