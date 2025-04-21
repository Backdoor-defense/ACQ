from torch.utils.data import DataLoader
from _logger import _LOGGER
from tqdm import tqdm
from utils.picker import *
from utils.controller import *
from main_config import dataset_config, model_config, auxiliary_config
import torch.cuda
import torch
import torch.nn.functional as F
from torchvision import transforms

class _TRAIN():
    def __init__(
        self,
        dataset_config: dict = dataset_config,
        model_config: dict = model_config,
        auxiliary_config: dict = auxiliary_config
    ) -> None:

        self.dataset_config = dataset_config
        self.model_config = model_config
        self.auxiliary_config = auxiliary_config
        self.attack = auxiliary_config['attack']

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.info()

    def info(self):
        dataset = self.dataset_config['dataset']
        model = self.model_config['model']
        train_batch_size = self.dataset_config['train_batch_size']
        self.epoch_num = self.model_config['epoch_num']

        optimizer = self.auxiliary_config['opt']
        learning_rate = self.auxiliary_config['lr']
        momentum = self.auxiliary_config['momentum']
        weight_decay = self.auxiliary_config['weight_decay']

        scheduler = self.auxiliary_config['stl']
        milestones = self.auxiliary_config['milestones']
        gamma = self.auxiliary_config['gamma']

        print('********************************** Training Information ***************************************')
        print('Dataset: {}\t\tmodel: {}\t\ttrain_batch_size: {}\t\tepoch number: {}'.format(dataset, model,
                                                                                            train_batch_size, self.epoch_num))
        print('Optimizer: {}\t\t\tlearning rate: {}\t\tmomentum: {}\t\t\tweight_decay: {}'.format(optimizer, learning_rate,
                                                                                                  momentum, weight_decay))
        print('Scheduler: {}\t\tmilestones: [{}, {}]\t\tgamma: {}'.format(scheduler, milestones[0],
                                                                          milestones[1], gamma))
        print('Device: {}'.format(self.device))

    
    def make_attacker(self):
        attacker = attack_picker(self.auxiliary_config)
        return attacker
    
    def auxiliary_maker(self, auxiliary_picker: callable = auxiliary_picker,
                        model_picker: callable = model_picker,
                        dataset_picker: callable = dataset_picker):

        model = model_picker(self.model_config)
        dataset_packer = dataset_picker(self.dataset_config)
        auxiliary = auxiliary_picker(self.auxiliary_config, model)

        self.train_size = dataset_packer['train_number']
        self.test_size = dataset_packer['test_number']

        trainset = dataset_packer['trainset']
        
        testset = dataset_packer['testset']

        optimizer = auxiliary['opt']
        scheduler = auxiliary['stl']

        train_attacker, test_attacker = self.make_attacker()
        bd_trainset = train_attacker.make_train_bddataset(trainset)
        if dataset_config['dataset'] == 'MNIST':
            trainset.transform = transforms.Compose([transforms.Resize(32),
                                                    transforms.Grayscale(3),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomCrop(32,4),
                                                    transforms.ToTensor()])
        else:
            trainset.transform = transforms.Compose([transforms.Resize(32),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomCrop(
                                                        32, 4),
                                                    transforms.ToTensor()])
        bd_testset = test_attacker.make_test_bddataset(testset)


        return model.to(self.device), bd_trainset, trainset, bd_testset, testset, optimizer, scheduler

    def testset_preprocess(self, testset:Dataset):
        img = testset.data[0]
        if len(img.shape) != 3:
            testset.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor()
                ]
            )
        else:
            testset.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor()
                ]
            )
        return testset

    def train(self):
        raise NotImplementedError

    def test(self, model: nn.Module, loss_func: nn.Module,
             testloader: DataLoader, bdloader: DataLoader, logger: _LOGGER):
        model.eval()
        logger.batch_logger_refresh()

        for _, (images, labels) in tqdm(enumerate(testloader), desc='Testing', colour='green', total=len(testloader)):
            images, labels = images.to(self.device), labels.to(self.device)
            batchsize = images.shape[0]

            with torch.no_grad():
                output: torch.Tensor = self._impl_output(model, images)
                clean_loss = loss_func(output, labels).item()

                pred = output.argmax(dim=1, keepdim=True)
                clean_num = pred.eq(labels.view_as(pred)).sum().item()

            logger.batch_acc_record_updater(clean_num, True)
            logger.batch_loss_record_updater(clean_loss, batchsize, True)

        for _, (images, labels) in tqdm(enumerate(bdloader), desc='BD Testing', colour='red', total=len(bdloader)):
            images, labels = images.to(self.device), labels.to(self.device)
            batchsize = images.shape[0]

            with torch.no_grad():
                output: torch.Tensor = self._impl_output(model, images)
                bd_loss = loss_func(output, labels).item()

                pred = output.argmax(dim=1, keepdim=True)
                targets = labels
                correct_num = pred.eq(targets.view_as(pred)).sum().item()

            logger.batch_acc_record_updater(correct_num, False)
            logger.batch_loss_record_updater(bd_loss, batchsize, False)

        logger.record_updater()
        logger.model_record_update(model)

    def epoch_results(self, logger: _LOGGER):
        clean_acc = logger.clean_acc_logger[-1]
        clean_loss = logger.clean_loss_logger[-1]
        print('Clean test acc: {}   Clean test loss: {}'.format(
            round(clean_acc, 2), round(clean_loss, 2)
        ))

        bd_acc = logger.bd_acc_logger[-1]
        bd_loss = logger.bd_loss_logger[-1]
        print('Attack: {}   Adv test acc: {}    Adv test loss: {}'.format(
                self.attack, round(bd_acc, 2), round(bd_loss, 2)
        ))

    def checkpoints_save(self, logger: _LOGGER):
        logger.make_checkpoints(self.attack)

    def model_updater(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _impl_output(self, model: nn.Module, images: torch.Tensor):
        output = model(images)

        if isinstance(output, torch.Tensor):
            pass

        elif isinstance(output, list):
            output = output[0]

        return output

    def make_loader(self, bd_trainset, bd_testset, testset):
        testset = self.testset_preprocess(testset)
        bd_trainloader = DataLoader(bd_trainset, self.dataset_config['train_batch_size'], shuffle=True)
        bd_testloader = DataLoader(bd_testset, batch_size = 1)
        testloader = DataLoader(testset, batch_size = 1)
        return bd_trainloader, bd_testloader, testloader

    def trained_model_eval(self, attack: str, model):
        _, _, _, bd_testset, testset, _, _ = self.auxiliary_maker()
        testset = self.testset_preprocess(testset)
        bd_test_loader = DataLoader(bd_testset, 500)
        clean_test_loader = DataLoader(testset, 500)

        test_size, bd_size = len(testset), len(bd_testset)
        loss_func = nn.CrossEntropyLoss()
        logger = _LOGGER(test_size, bd_size)

        save_path = logger.save_path
        model.load_state_dict(torch.load(
            save_path+'{}.pth'.format(attack))['state_dict'], strict=False)
        self.test(model, loss_func, clean_test_loader, bd_test_loader, logger)
        self.epoch_results(logger)

  