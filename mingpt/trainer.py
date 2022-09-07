"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.temperature = 2.0
        # teacher distillation parameters
        C.distil_scheduler = "linear" # "no": disabled; "linear": schedule it from provided value to 1-x
        C.alpha_distil = 0.99 # weight assigned to teacher loss; use 1-x for regular ce loss
        return C

    def __init__(self, config, model, train_dataset, **kwargs):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.teacher = kwargs.pop("teacher_model", None)
        self.iter_alpha_distil = config.alpha_distil

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device

        if self.teacher:
            self.teacher = self.teacher.to(self.device)
    
        self.model = self.model.to(self.device)
        print("running on device", self.device)
        # self.tea_loss = torch.tensor(0).to(self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    @staticmethod
    def _compute_linear_schedule(iter_num, max_iters, start_value, end_value):
        return iter_num / max_iters * (end_value - start_value) + start_value

    def run(self):
        model, config = self.model, self.config
        scaler = GradScaler()

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        while True:

            # Schedule alpha_distil for teacher distillation
            if self.teacher and config.distil_scheduler == "linear":
                self.iter_alpha_distil = Trainer._compute_linear_schedule(
                    iter_num=self.iter_num, max_iters=config.max_iters,
                    start_value=config.alpha_distil, end_value=1-config.alpha_distil
                )

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch
            # forward the model
            with autocast():
                logits, self.loss = model(x, y)

                if self.teacher is not None:
                    with torch.no_grad():
                        outputs_tea = self.teacher(x)
                        logits_tea = outputs_tea.logits

                    soft_logits = F.kl_div(
                            input=F.log_softmax(logits / config.temperature, dim=-1),
                            target=F.softmax(logits_tea / config.temperature, dim=-1),
                            reduction="batchmean",
                        ) * (config.temperature ** 2)

                    self.teacher_loss = soft_logits
                    self.ce_loss = self.loss
                    self.loss = self.iter_alpha_distil * soft_logits + (1-self.iter_alpha_distil) * self.loss


            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            scaler.scale(self.loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(self.optimizer)
            scaler.update()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break