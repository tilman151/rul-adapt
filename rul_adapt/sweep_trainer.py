import sys

sys.path.append('../')
import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions

from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, starting_logs, DictAsObject
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from utils import AverageMeter

from trainers.abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()
pl.seed_everything(42, workers=True)  # makes is reproducible

from trainers.train import Trainer

import argparse
parser = argparse.ArgumentParser()


import rul_datasets
import rul_adapt
import pytorch_lightning as pl
import omegaconf

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer





class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # sweep parameters
        self.num_sweeps = args.num_sweeps
        self.sweep_project_wandb = args.sweep_project_wandb
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def sweep(self):
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.da_method + '_' + self.backbone,
            'parameters': {**sweep_alg_hparams[self.da_method]}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)

        wandb.agent(sweep_id, self.train, count=sweep_runs_count)

    def train(self):
        run = wandb.init(config=self.hparams)
        self.hparams= wandb.config
        
        # To be known:
            # where to put # wandb_logger = WandbLogger()
            # How to log the outputs of the model to output frame
            # How to have generic construct that get the method and returns the construction
            # How to get the configs to enable model sweeping through 


        # Steps:
        # Data Loading step
        source = rul_datasets.CmapssReader(3)
        target = source.get_compatible(1, percent_broken=0.8)
        pre_dm = rul_datasets.RulDataModule(source, batch_size=32)
        dm = rul_datasets.DomainAdaptionDataModule( pre_dm, rul_datasets.RulDataModule(target, batch_size=32))

        # Backbone Construction step
        feature_extractor = rul_adapt.model.CnnExtractor(
            input_channels=14,
            conv_filters=[16, 16, 16],
            seq_len=30,
            fc_units=8, )
        regressor = rul_adapt.model.FullyConnectedHead(
            input_channels=8,
            units=[8, 1],
            act_func_on_last_layer=False,
        )
        domain_disc = rul_adapt.model.FullyConnectedHead(
            input_channels=8,
            units=[8, 1],
            act_func_on_last_layer=False,
        )

        # Logging 
        wandb_logger = WandbLogger(project = "<project_name>",)
        # Pretraining step 
        pre_approach = rul_adapt.approach.SupervisedApproach(
            lr=0.001, loss_type="mse", optim_type="adam", rul_scale=source.max_rul
        )
        pre_approach.set_model(feature_extractor, regressor)
        pre_trainer = pl.Trainer(max_epochs=1)
        pre_trainer.fit(pre_approach, pre_dm) # fixed this issue

        # Adaptation step
        approach = rul_adapt.approach.AdaRulApproach(
            lr=0.001,
            max_rul=source.max_rul,
            num_disc_updates=35,
            num_gen_updates=1,
        )
        approach.set_model(
            pre_approach.feature_extractor, pre_approach.regressor, domain_disc
        )
        trainer = pl.Trainer(max_epochs=1, logger = wandb_logger)
        trainer.fit(approach, dm)
        trainer.test(approach, dm)


        # for src_id, trg_id in self.dataset_configs.scenarios:
        #     for run_id in range(self.num_runs):
        #         # set random seed and create logger
        #         fix_randomness(run_id)
        #         self.logger, self.scenario_log_dir = starting_logs( self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id  )

        #         # average meters
        #         self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

        #         # load data and train model
        #         self.load_data(src_id, trg_id)

        #         # initiate the domain adaptation algorithm
        #         self.initialize_algorithm()

        #         # Train the domain adaptation algorithm
        #         self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)

        #         # calculate metrics and risks
        #         metrics = self.calculate_metrics()
        #         risks = self.calculate_risks()

        #         # append results to tables
        #         scenario = f"{src_id}_to_{trg_id}"
        #         table_results.add_data(scenario, run_id, *metrics)
        #         table_risks.add_data(scenario, run_id, *risks)

        # # calculate overall metrics and risks
        # total_results, summary_metrics = self.calculate_avg_std_wandb_table(table_results)
        # total_risks, summary_risks = self.calculate_avg_std_wandb_table(table_risks)

        # # log results to WandB
        # self.wandb_logging(total_results, total_risks, summary_metrics, summary_risks)

        # # finish the run
        # run.finish()



if __name__ == "__main__":

    # ========  Experiments Phase ================
    parser.add_argument('--phase',               default='test',         type=str, help='train, test')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name',               default='DDC_test',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='DDC',               type=str, help='NO_ADAPT, Deep_Coral, MMDA, DANN, CDAN, DIRT, DSAN, HoMM, CoDATS, AdvSKM, SASA, CoTMix, TARGET_ONLY')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default=r'../TS_Data',                  type=str, help='Path containing datase2t')
    parser.add_argument('--dataset',                default='HAR',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=5,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default= "cuda",                   type=str, help='cpu or cuda')

    # arguments
    args = parser.parse_args()

    # create trainier object
    trainer = Trainer(args)

    # train and test
    if args.phase == 'train':
        trainer.fit()
    elif args.phase == 'test':
        trainer.test()
