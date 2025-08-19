import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import socket

class DistributedForexTrainer:
    """Distributed training across your 3 machines"""
    
    def __init__(self):
        self.world_size = 3  # 3 machines
        self.master_addr = "192.168.1.100"  # Your GTX 1070 PC IP
        self.master_port = "12355"
        
    def setup_distributed(self, rank, world_size):
        """Initialize distributed training"""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        
        # Initialize process group
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                               rank=rank, world_size=world_size)
    
    def train_distributed_model(self, rank, world_size, model, dataset):
        """Train model across multiple machines"""
        self.setup_distributed(rank, world_size)
        
        # Wrap model with DDP
        if torch.cuda.is_available():
            model = model.cuda()
            ddp_model = DDP(model, device_ids=[rank])
        else:
            ddp_model = DDP(model)
        
        # Distributed data sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=128, sampler=sampler
        )
        
        # Training loop
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.0001)
        criterion = torch.nn.BCELoss()
        
        for epoch in range(500):  # Extended training
            sampler.set_epoch(epoch)
            
            for batch_idx, (data, target) in enumerate(dataloader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = criterion(output.squeeze(), target.float())
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0 and rank == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Cleanup
        dist.destroy_process_group()

# Machine-specific startup scripts:

# GTX 1070 PC (Rank 0 - Master):
def start_master_training():
    trainer = DistributedForexTrainer()
    mp.spawn(trainer.train_distributed_model, args=(3, model, dataset), nprocs=1)

# M1 MacBook 1 (Rank 1):
def start_worker1_training():
    trainer = DistributedForexTrainer()
    mp.spawn(trainer.train_distributed_model, args=(3, model, dataset), nprocs=1)

# M1 MacBook 2 (Rank 2):
def start_worker2_training():
    trainer = DistributedForexTrainer()
    mp.spawn(trainer.train_distributed_model, args=(3, model, dataset), nprocs=1)