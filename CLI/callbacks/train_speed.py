
from pytorch_lightning.callbacks import Callback
import time

class TrainingSpeedCallback(Callback):
    def __init__(self, log_every_n_steps=1):
        self.start_time = None
        self.epoch_start_time = None
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.epoch_start_time
        trainer.logger.log_metrics({"epoch_duration": epoch_duration}, step=trainer.current_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0:
            step_duration = time.time() - self.start_time
            trainer.logger.log_metrics({"step_duration": step_duration / (trainer.global_step + 1)}, step=trainer.global_step)
