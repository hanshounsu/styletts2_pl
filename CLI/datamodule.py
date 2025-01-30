
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from meldataset import FilePathDataset, Collater

from utils import get_data_path_list

class LJSpeechDataModule(pl.LightningDataModule):
    def __init__(self,
                train_path: str,
                val_path: str,
                root_path,
                OOD_path="Data/OOD_texts.txt",
                min_length=50,
                batch_size=4,
                device='cpu',):
        super().__init__()
        self.train_list, self.val_list = get_data_path_list(train_path, val_path)
        self.root_path = root_path
        self.OOD_data = OOD_path
        self.min_length = min_length
        self.batch_size = batch_size
        self.device = device
        
        self.collate_fn = Collater(return_wave=False)

    def setup(self, stage=None):
        self.train = FilePathDataset(
            data_list=self.train_list,
            root_path=self.root_path,
            OOD_data=self.OOD_data,
            min_length=self.min_length,
            validation=False,)

        self.val= FilePathDataset(
            data_list=self.val_list,
            root_path=self.root_path,
            OOD_data=self.OOD_data,
            min_length=self.min_length,
            validation=True,)


    def train_dataloader(self):
        return DataLoader(self.train,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=4,
                        drop_last=True,
                        collate_fn=self.collate_fn,
                        pin_memory=True,
                        # persistent_workers=True
                        )

    def val_dataloader(self):
        return DataLoader(self.val,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=4,
                        drop_last=False,
                        collate_fn=self.collate_fn,
                        pin_memory=True,
                        )

    # def test_dataloader(self):
    #     return DataLoader(self.val,
    #                       shuffle=False,
    #                       batch_size=self.batch_size,
    #                       num_workers=8,
    #                       sampler=None,
    #                       pin_memory=self.pin_memory)
