# main.py
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from CLI.module_2nd import Second
from CLI.datamodule import LJSpeechDataModule

# simple demo classes for your convenience
import os
import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import colored
from pytorch_lightning.profilers import AdvancedProfiler


class StyleTTS2CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("-d", "--debug", type=bool,
                            default=False, help="enable post-mortem debugging",)
        parser.add_argument("--wandb", type=bool,
                            default=False, help="wandb online/offline",)
        parser.add_argument("--batch_size", type=int, help="batch size",)
        parser.add_argument("--sample_rate", type=int, help="sample rate",)
        parser.add_argument("--test_save_path", type=str, help="test_save_path",)

    def before_fit(self):
        if not self.config.fit.ckpt_path:  # if not resuming from checkpoint
            self.now = id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        elif self.config.fit.ckpt_path:  # resuming from checkpoint
            ckpt_date = os.path.basename(
                os.path.dirname(self.config.fit.ckpt_path))
            print(colored("Continue training from checkpoint: ",
                  "red", attrs=['bold']), ckpt_date)
            self.now = id = ckpt_date
        # Logging
        wandb_logger = WandbLogger(save_dir=f"./logs/{self.now}",
                                   name=self.now,
                                   project="StyleTTS2",
                                   offline=(not self.config.fit.wandb),
                                   id=id)
        self.trainer.logger = wandb_logger

        # Model checkpoint (automatically called after validation)
        model_checkpoint_callback = ModelCheckpoint(
            dirpath=f'./checkpoints/{self.now}',
            monitor='',
            mode='min',
            save_top_k=20,
            save_last=True,
            verbose=True,
            save_on_train_epoch_end=True,
            filename='{step:07}-{val/diffusion_loss:.4f}')  # python recognized '/', '-' as '_'

        self.trainer.callbacks.append(model_checkpoint_callback)
        self.config.fit.test_save_path = f"./results/{self.now}"
        print(colored("Test results will be saved in: ", "green",
              attrs=['bold']), self.config.fit.test_save_path)

        # profiler = AdvancedProfiler(dirpath=f'./profiler/{self.now}', filename='profiler')
        # self.trainer.profiler = profiler


def cli_main():
    cli = StyleTTS2CLI(Second, LJSpeechDataModule,
                       save_config_kwargs={"overwrite": True},
                       )


if __name__ == "__main__":
    cli_main()