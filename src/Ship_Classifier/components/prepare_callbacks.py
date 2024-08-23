import os
import urllib.request as request
from zipfile import ZipFile
import torch 
import time
from torch.utils.tensorboard import SummaryWriter
from Ship_Classifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig,model, optimizer, loss):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
   
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return torch.utils.tensorboard.SummaryWriter(log_dir=tb_running_log_dir)  # Adjust as necessary to return the TensorBoard logger

    @property
    def _create_ckpt_callbacks(self):
       checkpoint_dir = self.config.checkpoint_model_filepath
       checkpoint = torch.save({
            'epoch': self.config.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
       }, checkpoint_dir)
       return checkpoint
    
    def get_tb_ckpt_callbacks(self):
        tb_callback = self._create_tb_callbacks
        ckpt_callback = self._create_ckpt_callbacks
        return tb_callback, ckpt_callback