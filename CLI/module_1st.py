from losses import *
from utils import *
from models import *
import torch.nn.functional as F
from torch import nn
import random
import torch
import numpy as np
import pytorch_lightning as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from Utils.PLBERT.util import load_plbert
import wandb


class First(pl.LightningModule):
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
                 prosodic_style_encoder: nn.Module,
                 audio_diffusion_conditional: nn.Module,
                 k_diffusion: nn.Module,
                 mpd: nn.Module,
                 msd: nn.Module,
                 wd: nn.Module,
                 optimizer: OptimizerCallable = torch.optim.AdamW,
                #  scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.OneCycleLR,
                 ):
        super().__init__()
        self.save_hyperparameters()

        # load pretrained models for text aligner and pitch extractor
        self.text_aligner, self.pitch_extractor = ASR, F0
        params_ASR, params_F0 = torch.load(ASR_checkpoint_path, map_location='cpu')['model'], torch.load(F0_checkpoint_path, map_location='cpu')['net']
        self.text_aligner.load_state_dict(params_ASR)
        self.pitch_extractor.load_state_dict(params_F0)
        self.n_down = 1

        self.plbert = load_plbert(PLBERT_dir)
        self.decoder = decoder
        self.text_encoder = text_encoder
        self.prosody_predictor = prosody_predictor
        self.acoustic_style_encoder = acoustic_style_encoder
        self.prosodic_style_encoder = prosodic_style_encoder
        self.k_diffusion = k_diffusion
        self.audio_diffusion_conditional = audio_diffusion_conditional
        self.msd = msd
        self.mpd = mpd
        # self.wd = wd

        self.optimizer = optimizer
        self.automatic_optimization = False

        self.stft_loss = MultiResolutionSTFTLoss()
        self.gl = GeneratorLoss(self.mpd, self.msd)
        self.dl = DiscriminatorLoss(self.mpd, self.msd)
        self.wl = WavLMLoss(slm['model'], 
                            wd,
                            model_sr=sr,
                            slm_sr=slm['sr'])
        
        self.loss_val = 0
        self.s2s_attn, self.en, self.gt, self.mel_input_length, self.waves = None, None, None, None, None
        self.global_step_ = 0

    def forward(self, x):
        return x


    # def on_after_backward(self) -> None:
    #     print("on_after_backward enter")
    #     for key, param in self.named_parameters():
    #         if param.grad is None:
    #             print(key)
    #     print("on_after_backward exit")

    def training_step(self, batch, batch_idx):
        optimizers, schedulers = {}, {}
        optimizers['text_encoder'], optimizers['acoustic_style_encoder'], optimizers['decoder'], optimizers['text_aligner'], optimizers['pitch_extractor'], optimizers['msd'], optimizers['mpd'] = self.optimizers()
        # schedulers['text_encoder'], schedulers['acoustic_style_encoder'], schedulers['decoder'], schedulers['text_aligner'], schedulers['pitch_extractor'], schedulers['msd'], schedulers['mpd'] = self.lr_schedulers()
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
            optimizers['msd'].step()#, schedulers['msd'].step()
            optimizers['mpd'].step()#, schedulers['mpd'].step()
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
            loss_slm = self.wl(wav, y_rec).mean()

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

        optimizers['text_encoder'].step()#, schedulers['text_encoder'].step()
        optimizers['acoustic_style_encoder'].step()#, schedulers['acoustic_style_encoder'].step()
        optimizers['decoder'].step()#, schedulers['decoder'].step()

        if self.current_epoch >= self.hparams.epochs['TMA_epoch']:
            optimizers['text_aligner'].step()#, schedulers['text_aligner'].step()
            optimizers['pitch_extractor'].step()#, schedulers['pitch_extractor'].step()
        
        # print('Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f',
        #       self.current_epoch+1, self.trainer.max_epochs, batch_idx+1, len(self.trainer.datamodule.train_dataloader()), loss_mel, loss_gen_all, d_loss, loss_mono, loss_s2s, loss_slm)
        self.trainer.logger.log_metrics({'train/mel_loss': loss_mel,
                                 'train/gen_loss': loss_gen_all,
                                 'train/d_loss': d_loss,
                                 'train/mono_loss': loss_mono,
                                 'train/s2s_loss': loss_s2s,
                                 'train/slm_loss': loss_slm,
                                 'train/text_encoder/lr': optimizers['text_encoder'].param_groups[0]['lr']}, step=self.global_step_)
        # self.log('train/mel_loss', loss_mel, on_step=True, logger=True)
        # self.log('train/gen_loss', loss_gen_all, on_step=True,  logger=True)
        # self.log('train/d_loss', d_loss, on_step=True, logger=True)
        # self.log('train/mono_loss', loss_mono, on_step=True, logger=True)
        # self.log('train/s2s_loss', loss_s2s, on_step=True, logger=True)
        # self.log('train/slm_loss', loss_slm, on_step=True, logger=True)
        # # log learning rate
        # self.log('train/text_encoder/lr', optimizers['text_encoder'].param_groups[0]['lr'], on_step=True, logger=True)
        self.global_step_ += 1
    
    def validation_step(self, batch, batch_idx):
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

        self.log('val/mel_loss', self.loss_val / len(self.trainer.datamodule.val_dataloader()), logger=True, sync_dist=True)

        # attn_image = get_image(self.s2s_attn[0].cpu().numpy().squeeze())
        # image shape has to be H,W,C(=1 if greyscale)
        # self.trainer.logger.log_image(key='val/attn', images=[self.s2s_attn[0].cpu().numpy()], caption=[str(self.global_step_)])
        # self.trainer.logger.experiment.log({
        #     'val/attn': wandb.Image(self.s2s_attn[0].cpu().numpy(), caption=str(self.global_step_)),
        # })
        self.trainer.logger.log_metrics({'val/attn' : [wandb.Image(self.s2s_attn[0].cpu().numpy(), caption=str(self.global_step_))]})

        for bib in range(len(self.en)):
            mel_length = int(self.mel_input_length[bib].item())
            gt = self.gt[bib, :, :mel_length].unsqueeze(0)
            en = self.en[bib, :, :mel_length // 2].unsqueeze(0)

            F0_real, _, _ = self.pitch_extractor(gt.unsqueeze(1))
            # F0_real = F0_real.unsqueeze(0)
            s = self.acoustic_style_encoder(gt.unsqueeze(1))
            real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
            y_rec = self.decoder(en, F0_real, real_norm, s)

            # self.trainer.logger.log_audio(key='val/y_' + str(bib), audios=[y_rec.squeeze().cpu().numpy()], sample_rate=[self.hparams.sr], caption=[str(self.global_step_)])
            # self.trainer.logger.experiment.log({
            #     'val/y_' + str(bib): wandb.Audio(y_rec.squeeze().cpu().numpy(), sample_rate=self.hparams.sr, caption=str(self.global_step_)),
            # })
            self.trainer.logger.log_metrics({'val/y_' + str(bib): [wandb.Audio(y_rec.squeeze().cpu().numpy(), sample_rate=self.hparams.sr, caption=str(self.global_step_))]})
            if self.current_epoch == 0:
                # self.trainer.logger.log_audio(key='val/gt_' + str(bib), audios=[self.waves[bib]], sample_rate=[self.hparams.sr], caption=[str(self.global_step_)])
                # self.trainer.logger.experiment.log({
                #     'val/gt_' + str(bib): wandb.Audio(gt.squeeze().cpu().numpy(), sample_rate=self.hparams.sr, caption=str(self.global_step_)),
                # })
                self.trainer.logger.log_metrics({'val/gt_' + str(bib): [wandb.Audio(self.waves[bib], sample_rate=self.hparams.sr, caption=str(self.global_step_))]})

            if bib >= 6: break

        self.loss_val, self.s2s_attn, self.en, self.gt, self.mel_input_length, self.waves = 0, None, None, None, None, None


    def configure_optimizers(self):
        optimizers = [self.optimizer(model.parameters()) for model in [self.text_encoder,
                                                      self.acoustic_style_encoder,
                                                      self.decoder,
                                                      self.text_aligner,
                                                      self.pitch_extractor,
                                                      self.msd,
                                                      self.mpd]]
        # schedulers = [torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=1.0e-4,
        #     steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
        #     epochs=self.trainer.max_epochs,
        #     pct_start=0.0,
        #     div_factor=1,
        #     final_div_factor=1,
        # ) for optimizer in optimizers]

        # return [{"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}} for optimizer, scheduler in zip(optimizers, schedulers)]
        return optimizers
