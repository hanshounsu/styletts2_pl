from logging import StreamHandler
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Modules.slmadv import SLMAdversarialLoss
from Utils.JDC.model import JDCNet
from Utils.ASR.models import ASRCNN
import traceback
from accelerate.logging import get_logger
import logging
from torch.utils.tensorboard import SummaryWriter
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import LoggerType
from accelerate import Accelerator
import time
from optimizers import build_optimizer
from losses import *
from utils import *
from meldataset import build_dataloader
from models import *
import librosa
import torchaudio
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from munch import Munch
import random
import warnings
import click
import torch
import numpy as np
import shutil
import yaml
import sys
import re
import os.path as osp
import os
import pytorch_lightning as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from Utils.PLBERT.util import load_plbert

import copy


class Second(pl.LightningModule):
    def __init__(self,
                 sr: int,
                 max_len: int,
                 multispeaker: bool,
                 epochs: dict,
                 loss_params: dict,
                 slm: dict,
                 ASR: nn.Module,
                 ASR_checkpoint_path: str,
                 F0: nn.Module,
                 F0_checkpoint_path: str,
                 PLBERT_dir: str,
                 decoder: nn.Module,
                 text_encoder: nn.Module,
                 prosody_predictor: nn.Module,
                 acoustic_style_encoder: nn.Module,
                #  prosodic_style_encoder: nn.Module,
                 audio_diffusion_conditional: nn.Module,
                 k_diffusion: nn.Module,
                 mpd: nn.Module,
                 msd: nn.Module,
                 wd: nn.Module,
                 slmadv_params: dict,
                 optimizer: OptimizerCallable = torch.optim.AdamW,
                 optimizer_bert: OptimizerCallable = torch.optim.AdamW,
                 optimizer_ft: OptimizerCallable = torch.optim.AdamW,
                #  scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.OneCycleLR,
                 ):
        super().__init__()
        self.save_hyperparameters()

        # load pretrained models for text aligner and pitch extractor
        self.text_aligner, self.pitch_extractor = ASR, F0
        
        # these will be loaded from first stage
        # params_ASR, params_F0 = torch.load(ASR_checkpoint_path, map_location='cpu')['model'], torch.load(F0_checkpoint_path, map_location='cpu')['net']
        # self.text_aligner.load_state_dict(params_ASR)
        # self.pitch_extractor.load_state_dict(params_F0) 
        self.n_down = self.text_aligner.n_down

        self.bert = load_plbert(PLBERT_dir)
        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, self.prosody_predictor.text_encoder.d_model)

        self.decoder = decoder
        self.text_encoder = text_encoder
        self.prosody_predictor = prosody_predictor
        self.acoustic_style_encoder = acoustic_style_encoder
        self.prosodic_style_encoder = copy.deepcopy(acoustic_style_encoder)
        self.k_diffusion = k_diffusion
        self.audio_diffusion_conditional = audio_diffusion_conditional
        self.msd = msd
        self.mpd = mpd
        # self.wd = wd

        self.optimizer, self.optimizer_ft, self.optimizer_bert = optimizer, optimizer_ft, optimizer_bert
        self.automatic_optimization = False

        self.stft_loss = MultiResolutionSTFTLoss()
        self.gl = GeneratorLoss(self.mpd, self.msd)
        self.dl = DiscriminatorLoss(self.mpd, self.msd)
        self.wl = WavLMLoss(slm['model'], 
                            wd,
                            model_sr=sr,
                            slm_sr=slm['sr'])
        
        self.audio_diffusion_conditional.diffusion = self.k_diffusion
        self.sampler = DiffusionSampler(
            self.audio_diffusion_conditional.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False
        )

        self.stft_loss = MultiResolutionSTFTLoss()

        self.slmadv = SLMAdversarialLoss([self.], self.wl, self.sampler,
                                         slmadv_params['min_len'],
                                         slmadv_params['max_len'],
                                         batch_percentage=slmadv_params['batch_percentage'],
                                         skip_update=slmadv_params['iter'],
                                         sig=slmadv_params['sig']
                                         )

        self.loss_val = 0
        self.s2s_attn, self.en, self.gt, self.mel_input_length, self.waves = None, None, None, None, None

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        optimizers = {}
        optimizers['text_encoder'], optimizers['acoustic_style_encoder'], optimizers['decoder'], optimizers['text_aligner'], optimizers['pitch_extractor'], optimizers['msd'], optimizers['mpd'] = self.optimizers()
        waves = batch[0]
        texts, input_lengths, _, _, mels, mel_input_length, _ = batch[1:]

        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2 ** self.n_down))
            text_mask = length_to_mask(input_lengths)
        
        ppgs, s2s_pred, s2s_attn = self.text_aligner(mels, mask, texts)

        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)

        with torch.no_grad():
            attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
            attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
            attn_mask = (attn_mask < 1)
        
        s2s_attn.masked_fill_(attn_mask, 0.0)

        with torch.no_grad():
            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length)
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
        
        # encode
        t_en = self.text_encoder(texts, input_lengths, text_mask)

        # 50% of chance of using monotonic version
        if bool(random.getrandbits(1)):
            asr = (t_en @ s2s_attn)
        else:
            asr = (t_en @ s2s_attn_mono)
        
        # get clips
        mel_input_length_all = self.all_gather(mel_input_length)  # for balanced load
        mel_len = min([int(mel_input_length_all.min() / 2 - 1), self.hparams.max_len // 2])
        # mel_len = min([int(mel_input_length.min() / 2 - 1), self.hparams.max_len // 2])
        mel_len_st = int(mel_input_length.min() / 2 - 1)

        en = []
        gt = []
        wav = []
        st = []

        for bib in range(len(mel_input_length)):
            mel_length = int(mel_input_length[bib].item() / 2)

            random_start = np.random.randint(0, mel_length - mel_len)
            en.append(asr[bib, :, random_start:random_start+mel_len])
            gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

            y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
            wav.append(torch.from_numpy(y).to(self.device))

            # style reference (better to be different from the GT)
            random_start = np.random.randint(0, mel_length - mel_len_st)
            st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])

        en = torch.stack(en)
        gt = torch.stack(gt)
        st = torch.stack(st)

        wav = torch.stack(wav).float()

        # clip too short to be used by the style encoder
        if gt.shape[-1] < 80:
            return
        
        with torch.no_grad():
            real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
            F0_real, _, _ = self.pitch_extractor(gt.unsqueeze(1))
        
        s = self.acoustic_style_encoder(st.unsqueeze(1) if self.hparams.multispeaker else gt.unsqueeze(1))

        y_rec = self.decoder(en, F0_real, real_norm, s)

        # discriminator loss
        if self.current_epoch >= self.hparams.epochs['TMA_epoch']:
            optimizers['msd'].zero_grad()
            optimizers['mpd'].zero_grad()
            d_loss = self.dl(wav.unsqueeze(1).float(), y_rec.detach()).mean()
            self.manual_backward(d_loss)
            optimizers['msd'].step()
            optimizers['mpd'].step()
        else:
            d_loss = 0
        
        # generator loss
        optimizers['text_encoder'].zero_grad()
        optimizers['acoustic_style_encoder'].zero_grad()
        optimizers['decoder'].zero_grad()
        optimizers['text_aligner'].zero_grad()
        optimizers['pitch_extractor'].zero_grad()

        loss_mel = self.stft_loss(y_rec.squeeze(), wav)

        if self.current_epoch >= self.hparams.epochs['TMA_epoch']:  # start TMA training
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            loss_gen_all = self.gl(wav.unsqueeze(1).float(), y_rec).mean()
            with torch.no_grad():
                loss_slm = self.wl(wav, y_rec).mean()
            # loss_slm = self.wl(wav.detach(), y_rec).mean()

            g_loss = self.hparams.loss_params['lambda_mel'] * loss_mel + \
                self.hparams.loss_params['lambda_mono'] * loss_mono + \
                self.hparams.loss_params['lambda_s2s'] * loss_s2s + \
                self.hparams.loss_params['lambda_gen'] * loss_gen_all + \
                self.hparams.loss_params['lambda_slm'] * loss_slm
            
        else:
            loss_s2s = 0
            loss_mono = 0
            loss_gen_all = 0
            loss_slm = 0
            g_loss = loss_mel

        self.manual_backward(g_loss)

        optimizers['text_encoder'].step()
        optimizers['acoustic_style_encoder'].step()
        optimizers['decoder'].step()

        if self.current_epoch >= self.hparams.epochs['TMA_epoch']:
            optimizers['text_aligner'].step()
            optimizers['pitch_extractor'].step()
        
        # print('Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f',
        #       self.current_epoch+1, self.trainer.max_epochs, batch_idx+1, len(self.trainer.datamodule.train_dataloader()), loss_mel, loss_gen_all, d_loss, loss_mono, loss_s2s, loss_slm)
        # self.logger.log_metrics({'train/mel_loss': loss_mel, 'train/gen_loss': loss_gen_all, 'train/d_loss': d_loss, 'train/mono_loss': loss_mono, 'train/s2s_loss': loss_s2s, 'train/slm_loss': loss_slm}, step=self.global_step)
        self.log('train/mel_loss', loss_mel, on_step=True, logger=True)
        self.log('train/gen_loss', loss_gen_all, on_step=True,  logger=True)
        self.log('train/d_loss', d_loss, on_step=True, logger=True)
        self.log('train/mono_loss', loss_mono, on_step=True, logger=True)
        self.log('train/s2s_loss', loss_s2s, on_step=True, logger=True)
        self.log('train/slm_loss', loss_slm, on_step=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        for optimizer in self.optimizers(): optimizer.zero_grad()

        waves = batch[0]
        texts, input_lengths, _, _, mels, mel_input_length, _ = batch[1:]

        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2 ** self.n_down))
            ppgs, s2s_pred, s2s_attn = self.text_aligner(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)
            
            text_mask = length_to_mask(input_lengths)
            attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
            attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
            attn_mask = (attn_mask < 1)
            s2s_attn.masked_fill_(attn_mask, 0.0)

        # encode
        t_en = self.text_encoder(texts, input_lengths, text_mask)

        asr = (t_en @ s2s_attn)

        # get clips
        # mel_input_length_all = self.all_gather(mel_input_length)  # for balanced load
        mel_len = min([int(mel_input_length.min() / 2 - 1), self.hparams.max_len // 2])

        en = []
        gt = []
        wav = []
        for bib in range(len(mel_input_length)):
            mel_length = int(mel_input_length[bib] / 2)

            random_start = np.random.randint(0, mel_length - mel_len)
            en.append(asr[bib, :, random_start:random_start+mel_len])
            gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
            y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
            wav.append(torch.from_numpy(y).to('cuda'))
        
        wav = torch.stack(wav).float()

        en = torch.stack(en)
        gt = torch.stack(gt)

        F0_real, _, F0 = self.pitch_extractor(gt.unsqueeze(1))
        s = self.acoustic_style_encoder(gt.unsqueeze(1))
        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
        y_rec = self.decoder(en, F0_real, real_norm, s)

        loss_mel = self.stft_loss(y_rec.squeeze(), wav)

        self.loss_val += self.all_gather(loss_mel).mean().item()
        if batch_idx == 0:
            self.s2s_attn, self.en, self.gt, self.mel_input_length, self.waves = s2s_attn, en, gt, mel_input_length, waves
    
    def on_validation_epoch_end(self):
        print('Epochs [%d/%d], Val Mel Loss: %.5f', self.current_epoch+1, self.trainer.max_epochs, self.loss_val / len(self.trainer.datamodule.val_dataloader()))
        self.log('val/mel_loss', self.loss_val / len(self.trainer.datamodule.val_dataloader()), logger=True, sync_dist=True)

        attn_image = get_image(self.s2s_attn[0].cpu().numpy().squeeze())
        self.logger.log_image(key='val/attn', images=[attn_image], caption=[self.current_epoch])

        for bib in range(len(self.en)):
            mel_length = int(self.mel_input_length[bib].item())
            gt = self.gt[bib, :, :mel_length].unsqueeze(0)
            en = self.en[bib, :, :mel_length // 2].unsqueeze(0)

            F0_real, _, _ = self.pitch_extractor(gt.unsqueeze(1))
            # F0_real = F0_real.unsqueeze(0)
            s = self.acoustic_style_encoder(gt.unsqueeze(1))
            real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
            y_rec = self.decoder(en, F0_real, real_norm, s)

            self.logger.log_audio(key='val/y_' + str(bib), audios=[y_rec.squeeze().cpu().numpy()], sample_rate=[self.hparams.sr], caption=[self.current_epoch])
            if self.current_epoch == 0:
                self.logger.log_audio(key='val/gt_' + str(bib), audios=[self.waves[bib].squeeze()], sample_rate=[self.hparams.sr], caption=[self.current_epoch])

            if bib >= 6: break

        self.loss_val, self.s2s_attn, self.en, self.gt, self.mel_input_length, self.waves = 0, None, None, None, None, None


    def configure_optimizers(self):
        optimizers = [self.optimizer(model.parameters()) for model in [self.text_encoder,
                                                      self.text_aligner,
                                                      self.pitch_extractor,
                                                      self.msd,
                                                      self.mpd]]
        
        optimizers_ft = [self.optimizer_ft(model.parameters()) for model in [self.acoustic_style_encoder,
                                                      self.decoder]]
        
        optimizers_bert = [self.optimizer_bert(model.parameters()) for model in [self.plbert]]
        
        schedulers = [torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.trainer.max_epochs,
            pct_start=0.0, div_factor=1, final_div_factor=1,
        ) for optimizer in optimizers]

        schedulers_ft_bert = [torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'] * 2,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.trainer.max_epochs,
            pct_start=0.0, div_factor=1, final_div_factor=1,
        ) for optimizer in optimizers_ft + optimizers_bert]

        optimizers += (optimizers_ft + optimizers_bert)
        schedulers += schedulers_ft_bert
    
        return [{"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}} for optimizer, scheduler in zip(optimizers, schedulers)]


# load packages
warnings.simplefilter('ignore')


# simple fix for dataparallel that allows access to class attributes

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
 
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
        clamp=False
    )

    n_down = model.text_aligner.n_down

    best_loss = float('inf')  # best test loss
    loss_train_record = list([])
    loss_test_record = list([])
    iters = 0

    criterion = nn.L1Loss()  # F0 loss (regression)
    torch.cuda.empty_cache()


    print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    start_ds = False

    running_std = []

    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(model, wl, sampler,
                                slmadv_params.min_len,
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter,
                                sig=slmadv_params.sig
                                )

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]

        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()

        if epoch >= diff_epoch:
            start_ds = True

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length //
                                      (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except:
                    continue

                mask_ST = mask_from_lens(
                    s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = (t_en @ s2s_attn_mono)

                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.acoustic_style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.prosodic_style_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.prosodic_style_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.acoustic_style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            gs = torch.stack(gs).squeeze()  # global acoustic styles
            # ground truth for denoiser
            s_trg = torch.cat([gs, s_dur], dim=-1).detach()

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)

                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.module.diffusion.sigma_data = s_trg.std(
                        axis=-1).mean().item()  # batch-wise std estimation
                    running_std.append(
                        model.diffusion.module.diffusion.sigma_data)

                if multispeaker:
                    s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                                      embedding=bert_dur,
                                      embedding_scale=1,
                                      features=ref,  # reference from the same speaker as the embedding
                                      embedding_mask_proba=0.1,
                                      num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(
                        1), embedding=bert_dur, features=ref).mean()  # EDM loss
                    # style reconstruction loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
                else:
                    s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                                      embedding=bert_dur,
                                      embedding_scale=1,
                                      embedding_mask_proba=0.1,
                                      num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion.module.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur).mean()  # EDM loss
                    # style reconstruction loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
            else:
                loss_sty = 0
                loss_diff = 0

            d, p = model.predictor(d_en, s_dur,
                                   input_lengths,
                                   s2s_attn_mono,
                                   text_mask)

            mel_len = min(
                int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en = []
            gt = []
            st = []
            p_en = []
            wav = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                y = waves[bib][(random_start * 2) *
                               300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])

            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()

            if gt.size(-1) < 80:
                continue

            s_dur = model.prosodic_style_encoder(st.unsqueeze(
                1) if multispeaker else gt.unsqueeze(1))
            s = model.acoustic_style_encoder(st.unsqueeze(
                1) if multispeaker else gt.unsqueeze(1))

            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1]
                                * 2, F0.shape[2], 1).squeeze()

                asr_real = model.text_aligner.get_feature(gt)

                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

                y_rec_gt = wav.unsqueeze(1)
                y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

                if epoch >= joint_epoch:
                    # ground truth from recording
                    wav = y_rec_gt  # use recording since decoder is tuned
                else:
                    # ground truth from reconstruction
                    wav = y_rec_gt_pred  # use reconstruction since decoder is fixed

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

            y_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            if start_ds:
                optimizer.zero_grad()
                d_loss = dl(wav.detach(), y_rec.detach()).mean()
                d_loss.backward()
                optimizer.step('msd')
                optimizer.step('mpd')
            else:
                d_loss = 0

            # generator loss
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav)
            if start_ds:
                loss_gen_all = gl(wav, y_rec).mean()
            else:
                loss_gen_all = 0
            loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1],
                                      _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(
                    _s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            g_loss = loss_params.lambda_mel * loss_mel + \
                loss_params.lambda_F0 * loss_F0_rec + \
                loss_params.lambda_ce * loss_ce + \
                loss_params.lambda_norm * loss_norm_rec + \
                loss_params.lambda_dur * loss_dur + \
                loss_params.lambda_gen * loss_gen_all + \
                loss_params.lambda_slm * loss_lm + \
                loss_params.lambda_sty * loss_sty + \
                loss_params.lambda_diff * loss_diff

            running_loss += loss_mel.item()
            g_loss.backward()
            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()

            optimizer.step('bert_encoder')
            optimizer.step('bert')
            optimizer.step('predictor')
            optimizer.step('prosodic_style_encoder')

            if epoch >= diff_epoch:
                optimizer.step('diffusion')

            if epoch >= joint_epoch:
                optimizer.step('acoustic_style_encoder')
                optimizer.step('decoder')

                # randomly pick whether to use in-distribution text
                if np.random.rand() < 0.5:
                    use_ind = True
                else:
                    use_ind = False

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts

                slm_out = slmadv(i,
                                 y_rec_gt,
                                 y_rec_gt_pred,
                                 waves,
                                 mel_input_length,
                                 ref_texts,
                                 ref_lengths, use_ind, s_trg.detach(), ref if multispeaker else None)

                if slm_out is None:
                    continue

                d_loss_slm, loss_gen_lm, y_pred = slm_out

                # SLM generator loss
                optimizer.zero_grad()
                loss_gen_lm.backward()

                # compute the gradient norm
                total_norm = {}
                for key in model.keys():
                    total_norm[key] = 0
                    parameters = [p for p in model[key].parameters(
                    ) if p.grad is not None and p.requires_grad]
                    for p in parameters:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm[key] += param_norm.item() ** 2
                    total_norm[key] = total_norm[key] ** 0.5

                # gradient scaling
                if total_norm['predictor'] > slmadv_params.thresh:
                    for key in model.keys():
                        for p in model[key].parameters():
                            if p.grad is not None:
                                p.grad *= (1 / total_norm['predictor'])

                for p in model.predictor.duration_proj.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                for p in model.predictor.lstm.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                for p in model.diffusion.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                optimizer.step('bert_encoder')
                optimizer.step('bert')
                optimizer.step('predictor')
                optimizer.step('diffusion')

                # SLM discriminator loss
                if d_loss_slm != 0:
                    optimizer.zero_grad()
                    d_loss_slm.backward(retain_graph=True)
                    optimizer.step('wd')

            else:
                d_loss_slm, loss_gen_lm = 0, 0

            iters = iters + 1

            if (i+1) % log_interval == 0:
                logger.info('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f'
                            % (epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, d_loss, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty, loss_diff, d_loss_slm, loss_gen_lm))

                writer.add_scalar('train/mel_loss',
                                  running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/slm_loss', loss_lm, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/diff_loss', loss_diff, iters)
                writer.add_scalar('train/d_loss_slm', d_loss_slm, iters)
                writer.add_scalar('train/gen_loss_slm', loss_gen_lm, iters)

                running_loss = 0

                print('Time elasped:', time.time()-start_time)

        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    with torch.no_grad():
                        mask = length_to_mask(
                            mel_input_length // (2 ** n_down)).to('cuda')
                        text_mask = length_to_mask(
                            input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(
                            s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(
                            texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.prosodic_style_encoder(
                            mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.acoustic_style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze()
                    gs = torch.stack(gs).squeeze()
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(
                        texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                    d, p = model.predictor(d_en, s,
                                           input_lengths,
                                           s2s_attn_mono,
                                           text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []
                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(
                            0, mel_length - mel_len)
                        en.append(
                            asr[bib, :, random_start:random_start+mel_len])
                        p_en.append(
                            p[bib, :, random_start:random_start+mel_len])

                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                        y = waves[bib][(random_start * 2) *
                                       300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.prosodic_style_encoder(gt.unsqueeze(1))

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length-1],
                                              _text_input[1:_text_length-1])

                    loss_dur /= texts.size(0)

                    s = model.acoustic_style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except Exception as e:
                    print(f"run into exception", e)
                    traceback.print_exc()
                    continue

        print('Epochs:', epoch + 1)
        logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (
            loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n\n\n')
        print('\n\n\n')
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_align / iters_test, epoch + 1)
        writer.add_scalar('eval/F0_loss', loss_f / iters_test, epoch + 1)

        if epoch < joint_epoch:
            # generating reconstruction examples with GT duration

            with torch.no_grad():
                for bib in range(len(asr)):
                    mel_length = int(mel_input_length[bib].item())
                    gt = mels[bib, :, :mel_length].unsqueeze(0)
                    en = asr[bib, :, :mel_length // 2].unsqueeze(0)

                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                    F0_real = F0_real.unsqueeze(0)
                    s = model.acoustic_style_encoder(gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                    y_rec = model.decoder(en, F0_real, real_norm, s)

                    writer.add_audio(
                        'eval/y' + str(bib), y_rec.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    s_dur = model.prosodic_style_encoder(gt.unsqueeze(1))
                    p_en = p[bib, :, :mel_length // 2].unsqueeze(0)

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

                    y_pred = model.decoder(en, F0_fake, N_fake, s)

                    writer.add_audio(
                        'pred/y' + str(bib), y_pred.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    if epoch == 0:
                        writer.add_audio(
                            'gt/y' + str(bib), waves[bib].squeeze(), epoch, sample_rate=sr)

                    if bib >= 5:
                        break
        else:
            # generating sampled speech from text directly
            with torch.no_grad():
                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.acoustic_style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.prosodic_style_encoder(ref_mels.unsqueeze(1))
                    ref_s = torch.cat([ref_ss, ref_sp], dim=1)

                for bib in range(len(d_en)):
                    if multispeaker:
                        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                                         embedding=bert_dur[bib].unsqueeze(0),
                                         embedding_scale=1,
                                         # reference from the same speaker as the embedding
                                         features=ref_s[bib].unsqueeze(0),
                                         num_steps=5).squeeze(1)
                    else:
                        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                                         embedding=bert_dur[bib].unsqueeze(0),
                                         embedding_scale=1,
                                         num_steps=5).squeeze(1)

                    s = s_pred[:, 128:]
                    ref = s_pred[:, :128]

                    d = model.predictor.text_encoder(d_en[bib, :, :input_lengths[bib]].unsqueeze(0),
                                                     s, input_lengths[bib, ...].unsqueeze(0), text_mask[bib, :input_lengths[bib]].unsqueeze(0))

                    x, _ = model.predictor.lstm(d)
                    duration = model.predictor.duration_proj(x)

                    duration = torch.sigmoid(duration).sum(axis=-1)
                    pred_dur = torch.round(duration.squeeze()).clamp(min=1)

                    pred_dur[-1] += 5

                    pred_aln_trg = torch.zeros(
                        input_lengths[bib], int(pred_dur.sum().data))
                    c_frame = 0
                    for i in range(pred_aln_trg.size(0)):
                        pred_aln_trg[i, c_frame:c_frame +
                                     int(pred_dur[i].data)] = 1
                        c_frame += int(pred_dur[i].data)

                    # encode prosody
                    en = (d.transpose(-1, -2) @
                          pred_aln_trg.unsqueeze(0).to(texts.device))
                    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
                    out = model.decoder((t_en[bib, :, :input_lengths[bib]].unsqueeze(0) @ pred_aln_trg.unsqueeze(0).to(texts.device)),
                                        F0_pred, N_pred, ref.squeeze().unsqueeze(0))

                    writer.add_audio(
                        'pred/y' + str(bib), out.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    if bib >= 5:
                        break

        if epoch % saving_epoch == 0:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model},
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)

            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(
                    np.mean(running_std))

                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

if __name__ =="__main__":
    main()
