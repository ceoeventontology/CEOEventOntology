# Created by ceoeventontology at 2023/2/8
"""
improve embedding with external event data from wordnet
"""
import os
import yaml
import pickle
from copy import deepcopy
import numpy as np
import random
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def _extract_features(samples, emb_source, emb_merge_format):
    X = dict()

    for id, sample in samples.items():
        # NOTE: some ids are ignored if the gt predicate is correctly predicted by SRL model
        sample = samples[id]
        cur_X = list()
        for one_source in emb_source:
            cur_emb = sample[one_source]
            if cur_emb is None:
                continue
            cur_X.append(cur_emb)
        if len(cur_X) == 0:
            # all emb sources are none
            X[id] = sample['sent_emb']
        else:
            if emb_merge_format == 'average':
                X[id] = np.mean(np.stack(cur_X, axis=0), axis=0)
            elif emb_merge_format == 'concat':
                X[id] = np.concatenate(cur_X)

    return X

class EventDataset(torch_data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class PairDataset(LightningDataModule):
    def __init__(self, config, res_folder, pin_memory=False):
        super().__init__()
        self.depth = config['exp_params']['depth']
        self.train_batch_size, self.val_batch_size = config['data_params']['train_batch_size'], config['data_params']['val_batch_size']
        self.num_workers = config['data_params']['num_workers']
        self.pin_memory = pin_memory
        seed = config['exp_params']['manual_seed']
        self.emb_source = sorted(config['data_params']['emb_source'].split(','))
        self.emb_merge_format = config['data_params']['emb_merge_format']
        self.res_folder = res_folder
        seed_everything(seed, True)

    def setup(self, stage=None):
        feature_saved_path = f'{self.res_folder}/plain_embedding/features_embedding.tsv'
        features_ = np.loadtxt(feature_saved_path, dtype=np.float32)
        self.test_dataset = EventDataset(features_)

        print(f'load {len(self.test_dataset)} val samples')

        return

    def test_dataloader(self):
        return torch_data.DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

class autoencoder(nn.Module):
    def __init__(self, in_channels, layers, dropout, loss_name, regularization_flag, regularization_weight, margin, tau):
        super().__init__()
        modules = list()
        in_channels_ = deepcopy(in_channels)
        for h_dim in layers:
            modules.append(nn.Sequential(
                nn.Linear(in_channels_, h_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(p=dropout)
            ))
            in_channels_ = h_dim
        self.encoder = nn.Sequential(*modules)

        reversed_layer = layers[::-1]
        modules = list()
        for i in range(len(reversed_layer)-1):
            modules.append(nn.Sequential(
                nn.Linear(reversed_layer[i], reversed_layer[i+1]),
                nn.LeakyReLU(),
                nn.BatchNorm1d(reversed_layer[i+1]),
                nn.Dropout(p=dropout)
            ))
        modules.append(nn.Linear(reversed_layer[-1], in_channels))
        self.decoder = nn.Sequential(*modules)

        self.loss_name = loss_name
        self.regularization_flag = regularization_flag
        self.regularization_weight = regularization_weight
        self.margin = margin
        self.tau = tau
        assert self.loss_name == 'triplet'
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, input):
        en_output = self.encoder(input)
        de_output = self.decoder(en_output)
        return en_output, de_output

    def loss_function(self, input, en_output, de_output):
        batch_size = input.shape[0]
        recons_loss = F.mse_loss(de_output, input)
        wordnet_en_output = en_output[int(0.5*batch_size):]
        anchor = wordnet_en_output[::2]
        positive = wordnet_en_output[1::2]

        if self.loss_name == 'triplet':
            negative = torch.flip(positive, dims=[0])
            sim_loss = self.loss_fn(anchor, positive, negative)
        else:
            sim_loss = self.loss_fn(anchor, positive)
        if self.regularization_flag:
            loss = recons_loss + self.regularization_weight * sim_loss
        else:
            loss = recons_loss

        return {'loss': loss, 'reconstruction': recons_loss, 'similarity': sim_loss}

class Experiment(pl.LightningModule):
    def __init__(self, params, log_dir):
        super().__init__()

        self.model = autoencoder(
            in_channels=params['model_params']['in_channels'],
            layers=params['model_params']['layers'],
            dropout=params['exp_params']['dropout'],
            loss_name=params['exp_params']['loss_name'],
            regularization_flag=params['exp_params']['regularization_flag'],
            regularization_weight=params['exp_params']['regularization_weight'],
            margin=params['exp_params']['margin'],
            tau=params['exp_params']['tau']
        )
        self.params = params
        self.seed = self.params['exp_params']['manual_seed']
        self.log_dir = log_dir

    def forward(self, input):
        return self.model(input)

    def test_step(self, batch, batch_idx):
        en_output, _ = self.forward(batch[:, 2:])
        return batch[:, :2], en_output

    def test_epoch_end(self, outputs):
        Y, X = list(), list()
        for label_info, feature in outputs:
            Y.append(label_info[:, 1])
            X.append(feature)
        X = torch.cat(X, dim=0).cpu().detach().numpy()
        Y = torch.cat(Y, dim=0).cpu().detach().numpy().astype(int)

        filename = os.path.join(self.log_dir, 'features_embedding.tsv')
        fout = open(filename, 'w')
        for id, (x, y) in tqdm(enumerate(zip(X, Y)), total=len(X)):
            x_str = '\t'.join(list(map(str, x)))
            fout.write(str(id))
            fout.write('\t')
            fout.write(str(y))
            fout.write('\t')
            fout.write(x_str)
            fout.write('\n')
        fout.close()
        print(f'save learned embeddings at {filename}')

        return

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['exp_params']['LR'],
                               weight_decay=self.params['exp_params']['weight_decay'])
        optims.append(optimizer)


        scheduler = optim.lr_scheduler.ExponentialLR(
            optims[0],
            gamma=self.params['exp_params']['scheduler_gamma'])
        scheds.append(scheduler)

        return optims, scheds

def run_embedding(config, model_folder, saved_dir):
    data = PairDataset(config,
                       res_folder=os.path.dirname(saved_dir),
                       pin_memory=len(config['trainer_params']['gpus']) != 0
                       )
    data.setup()

    runner = Trainer(
        logger=False,
        callbacks=[],
        **config['trainer_params']
    )
    experiment = Experiment(params=config, log_dir=saved_dir)
    print(f'save log at {saved_dir}')

    print(f"======= Testing =======")
    if os.path.exists(os.path.join(model_folder, 'config.yaml')):
        with open(os.path.join(model_folder, 'config.yaml'), 'r') as f:
            # old_config = yaml.safe_load(f)
            old_config = yaml.load(f, Loader=yaml.Loader)
        old_device, new_device = old_config['trainer_params']['gpus'], config['trainer_params']['gpus']
        if old_device[0] == -1:
            old_device_name = 'cpu'
        else:
            old_device_name = f'cuda:{old_device[0]}'
        if new_device[0] == -1:
            new_device_name = 'cpu'
        else:
            new_device_name = f'cuda:{new_device[0]}'
    else:
        old_device_name, new_device_name = 'cuda:0', 'cuda:0'

    checkpoint_path = os.listdir(os.path.join(model_folder, "checkpoints"))
    assert len(checkpoint_path) == 1
    print(f'load from {os.path.join(model_folder, "checkpoints", checkpoint_path[0])}')
    experiment = experiment.load_from_checkpoint(
        checkpoint_path=os.path.join(model_folder, 'checkpoints', checkpoint_path[0]),
        # hparams_file=os.path.join(tb_logger.log_dir, 'hparams.yaml'),
        map_location={old_device_name: new_device_name},
        params=config,
        log_dir=saved_dir
    )
    runner.test(experiment, datamodule=data)

    return
