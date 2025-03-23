import torch
import os
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import OrderedDict
from engine.optimizer import make_optimizer
from utils.misc import AverageMeter, EarlyStop
from utils.setup_logging import get_logger
from timm.utils import accuracy, update_summary
import numpy as np
logger = get_logger("Prompt_CAM")
torch.backends.cudnn.benchmark = False

class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """

    def __init__(
            self,
            model, tune_parameters, params
    ) -> None:
        self.params = params
        self.model = model
        self.device = params.device
        self.cls_criterion = nn.CrossEntropyLoss()

        if 'test_data' not in params:
            # solver related
            logger.info("\tSetting up the optimizer...")
            self.optimizer = make_optimizer(tune_parameters, params)
            self.scheduler = CosineLRScheduler(self.optimizer, t_initial=params.epoch,
                                               warmup_t=params.warmup_epoch, lr_min=params.lr_min,
                                               warmup_lr_init=params.warmup_lr_init)
            self.total_epoch = self.params.epoch
            if self.params.early_patience > 0:
                self.early_stop_check = EarlyStop(self.params.early_patience)

    def forward_one_batch(self, samples, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            samples
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        samples = samples.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        # forward
        with torch.set_grad_enabled(is_train):
            outputs,_ = self.model(samples)  # (batchsize, num_cls)
            if self.params.train_type == 'prompt_cam':
                outputs = outputs.squeeze(-1)
            loss = self.cls_criterion(outputs, targets)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1, (-1, -1)
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1, (-1, -1)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            #for name, param in self.model.named_parameters():
            #    if param.grad is not None:
            #        print(f"After backward: {name} has grad with mean {param.grad.mean().item()}")
    
            self.optimizer.step()

        return loss, outputs, (acc1, acc5)

    def train_one_epoch(self, epoch, loader):
        loss_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        lr = self.scheduler._get_lr(epoch)
        logger.info(
            "Training {} / {} epoch, with learning rate {}".format(
                epoch + 1, self.total_epoch, lr
            )
        )
        # Enable training mode
        self.model.train()

        num_updates = epoch * len(loader)
        for idx, (samples, targets) in enumerate(loader):
            train_loss, _, (acc1, acc5) = self.forward_one_batch(samples, targets, True)
            if not isinstance(train_loss, int):
                loss_m.update(train_loss.item(), samples.shape[0])
                top1_m.update(acc1.item(), samples.shape[0])
                top5_m.update(acc5.item(), samples.shape[0])
            del train_loss, acc1, acc5, _, samples, targets
            num_updates += 1
            self.scheduler.step_update(num_updates=num_updates, metric=loss_m.avg)

        logger.info(
            "Epoch {} / {}: ".format(epoch + 1, self.total_epoch)
            + "average train loss: {:.2f}, ".format(loss_m.avg)
            + "average train top1: {:.2f} ".format(top1_m.avg)
            + "average train top5: {:.2f}".format(top5_m.avg))

        return OrderedDict(
            [('loss', round(loss_m.avg, 2)), ('top1', round(top1_m.avg, 2)), ('top5', round(top5_m.avg, 2))])

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """

        for epoch in range(self.total_epoch):
            train_metrics = self.train_one_epoch(epoch, train_loader)

            if (epoch % self.params.eval_freq == 0) or epoch == self.total_epoch - 1:
                if test_loader is not None:
                    eval_metrics = self.eval_classifier(
                        test_loader, "test")
                elif val_loader is not None:
                    eval_metrics = self.eval_classifier(val_loader, "val")
                else:
                    raise Exception('Both val and test loaders are missing. ')

                if self.params.early_patience > 0:
                    stop, save_model = self.early_stop_check.early_stop(eval_metrics)
                    if save_model and self.params.store_ckp:
                        torch.save({'model_state_dict': self.model.state_dict()},
                                   os.path.join(self.params.output_dir, 'model.pt'))
                    if stop:
                        return train_metrics, self.early_stop_check.max_metrics, eval_metrics
                if self.params.debug:
                    update_summary(
                        epoch, train_metrics, eval_metrics, os.path.join(self.params.output_dir, 'summary.csv'),
                        write_header=epoch == 0)
            self.scheduler.step(epoch)

        if self.params.store_ckp and not os.path.isfile(os.path.join(self.params.output_dir, 'model.pt')):
            torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.params.output_dir, 'model.pt'))
        return train_metrics, self.early_stop_check.max_metrics, eval_metrics

    @torch.no_grad()
    def eval_classifier(self, loader, prefix):
        """evaluate classifier"""

        loss_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        # Enable eval mode
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(loader):
                loss, outputs, (acc1, acc5) = self.forward_one_batch(samples, targets, False)
                if not isinstance(loss, int):
                    loss_m.update(loss.item(), samples.shape[0])
                    top1_m.update(acc1.item(), samples.shape[0])
                    top5_m.update(acc5.item(), samples.shape[0])
                del loss, outputs, acc1, acc5
        logger.info(
            f"Inference ({prefix}):"
            + "average loss: {:.2f}, ".format(loss_m.avg)
            + "average top1: {:.2f} ".format(top1_m.avg)
            + "average top5: {:.2f}".format(top5_m.avg))
        return OrderedDict(
            [('loss', round(loss_m.avg, 2)), ('top1', round(top1_m.avg, 2)), ('top5', round(top5_m.avg, 2))])

    def load_weight(self):
        self.model.load_state_dict(torch.load(self.params.output_dir + '/model.pt')['model_state_dict'])

    @torch.no_grad()
    def collect_logits(self, loader):
        self.model.eval()
        all_logits = []
        gt = []
        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(loader):
                loss, outputs, (acc1, acc5) = self.forward_one_batch(samples, targets, False)
                all_logits.append(outputs.cpu().detach().numpy())
                gt.append(targets.cpu().detach().numpy())
        return np.concatenate(all_logits, axis=0), np.concatenate(gt, axis=0)