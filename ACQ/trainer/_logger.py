from main_config import dataset_config, model_config
from utils.controller import *
import torch.nn as nn
import torch


class _LOGGER():
    def __init__(self, test_size: int, bd_size: int) -> None:
        self.clean_loss_logger = list()
        self.clean_acc_logger = list()

        self.bd_loss_logger = list()
        self.bd_acc_logger = list()

        self.test_size = test_size
        self.bd_size = bd_size
        self.makefile()

    def batch_logger_refresh(self):
        self.batch_clean_acc = list()
        self.batch_clean_loss = list()

        self.batch_bd_acc = list()
        self.batch_bd_loss = list()

    def record_updater(self):
        self.acc_record_updater()
        self.loss_record_updater()
        self.batch_logger_refresh()

    def batch_acc_record_updater(self, correct_num: int, clean: bool):
        if clean:
            self.batch_clean_acc.append(correct_num)
        else:
            self.batch_bd_acc.append(correct_num)

    def batch_loss_record_updater(self, avg_loss: float, batchsize: int, clean: bool):
        if clean:
            self.batch_clean_loss.append(avg_loss*batchsize)
        else:
            self.batch_bd_loss.append(avg_loss*batchsize)

    def makefile(self, dataset: str = dataset_config['dataset'],
                 model: str = model_config['model'],
                 save_path: str = model_config['save_path']):

        self.save_path = save_path + '{}/{}/'.format(dataset, model)
        makedir(self.save_path)
        print('Save path created.')

    def acc_record_updater(self):
        avg_clean_acc = sum(self.batch_clean_acc)/self.test_size
        avg_clean_acc *= 100
        self.clean_acc_logger.append(avg_clean_acc)

        avg_bd_acc = sum(self.batch_bd_acc)/self.bd_size
        avg_bd_acc *= 100
        self.bd_acc_logger.append(avg_bd_acc)

    def loss_record_updater(self):
        avg_clean_loss = sum(self.batch_clean_loss)/self.test_size
        self.clean_loss_logger.append(avg_clean_loss)

        avg_bd_loss = sum(self.batch_bd_loss)/self.bd_size
        self.bd_loss_logger.append(avg_bd_loss)

    def model_record_update(self, model: nn.Module):
        self.state_dict = model.state_dict()

    def make_checkpoints(self, attack: str):
        ckpt = {
            'clean_loss': self.clean_loss_logger,
            'clean_acc': self.clean_acc_logger,
            'bd_loss': self.bd_loss_logger,
            'bd_acc': self.bd_acc_logger,
            'state_dict': self.state_dict
        }
        torch.save(ckpt, self.save_path+'{}.pth'.format(attack))
