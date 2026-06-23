import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
import lightning as L
from torchmetrics import Accuracy
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
import os
from argparse import ArgumentParser

from models import DERFormer
from utils import param_groups_weight_decay


class FERModel(L.LightningModule):
    def __init__(
        self,
        embed_dim=512,
        depth=20,
        num_heads=8,
        mlp_ratio=4,
        init_values=1e-5,
        reg_tokens=4,
        pos_drop_rate=0.,
        patch_drop_rate=0.,
        drop_path_rate=0.,
        head_drop_rate=0.,
        mixup_alpha=1.0,
        mixup_prob=0.5,
        num_classes=6,
        label_smoothing=0.1,
        max_epochs=100,
        batch_size=32,
        lr=1e-3,
        wd=5e-4,
        momentum=0.9,
        cache_root=None
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.cache_root = cache_root
        
        self.model = DERFormer(
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            init_values,
            reg_tokens,
            pos_drop_rate,
            patch_drop_rate,
            drop_path_rate,
            head_drop_rate,
            num_classes,
        )
        
        self.mixup = transforms.RandomApply(nn.ModuleList([
            transforms.MixUp(alpha=mixup_alpha, num_classes=num_classes)
        ]), p=mixup_prob)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.macro_acc = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
    
    def forward(self, x):
        return self.model(x)
    
    @torch.no_grad()
    def predict(self, x):
        return self(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = self.mixup(x, y)
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def get_metrics(self, y_pred, y):
        self.acc.update(y_pred, y)
        self.macro_acc.update(y_pred, y)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        self.get_metrics(y_pred, y)
    
    def on_test_epoch_end(self):
        self.log('test_acc', self.acc.compute(), prog_bar=True)
        self.log('test_macro_acc', self.macro_acc.compute(), prog_bar=True)
        self.acc.reset()
        self.macro_acc.reset()
        return super().on_test_epoch_end()
    
    def configure_optimizers(self):
        param_groups = param_groups_weight_decay(
            self,
            self.wd,
            ['token', 'pos_embed']
        )
        optimizer = optim.SGD(param_groups, lr=self.lr, momentum=self.momentum, nesterov=True)
        dataset_length = sum(len(files) for _, _, files in os.walk(os.path.join(self.cache_root, 'train')))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.lr,
            dataset_length // self.batch_size * self.max_epochs,
            pct_start=0.1,
            final_div_factor=1e6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.RandomRotation(degrees=5),
            transforms.RandomPhotometricDistort(),
            transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
            transforms.RandomErasing(scale=(0.02, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        dataset = ImageFolder(
            os.path.join(self.cache_root, 'train'),
            transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        return dataloader
    
    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize(112),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        dataset = ImageFolder(
            os.path.join(self.cache_root, 'test'),
            transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )
        return dataloader
    

def train(args, fold):
    L.seed_everything(42)
    
    model = FERModel(
        cache_root=os.path.join(args.cache_root, str(fold)),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        init_values=args.init_values,
        reg_tokens=args.reg_tokens,
        pos_drop_rate=args.pos_drop_rate,
        patch_drop_rate=args.patch_drop_rate,
        drop_path_rate=args.drop_path_rate,
        head_drop_rate=args.head_drop_rate,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        num_classes=args.num_classes,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        momentum=args.momentum,
    )
    trainer = L.Trainer(
        max_epochs=model.max_epochs,
        callbacks=[
            ModelSummary(max_depth=2),
            ModelCheckpoint(
                save_top_k=0,
                save_last=True, 
            ),
        ],
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(model)
    trainer.test(model, ckpt_path='last')


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--cache-root', type=str, required=True)
    
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--init-values', type=float, default=1e-5)
    parser.add_argument('--reg-tokens', type=int, default=4)
    
    parser.add_argument('--pos-drop-rate', type=float, default=0.)
    parser.add_argument('--patch-drop-rate', type=float, default=0.)
    parser.add_argument('--drop-path-rate', type=float, default=0.)
    parser.add_argument('--head-drop-rate', type=float, default=0.)
    
    parser.add_argument('--mixup-alpha', type=float, default=1.)
    parser.add_argument('--mixup-prob', type=float, default=0.5)
    
    parser.add_argument('--num-classes', type=int, default=7)
    
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()

    for fold in range(4):
        train(args, fold)
