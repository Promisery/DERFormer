import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.v2.functional import normalize
from timm.layers import Mlp

from .iresnet import IResNet
from .vit import ViT


class Enhancer(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        mlp_ratio=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, 2*embed_dim)
        
        self.mlp = Mlp(embed_dim, int(embed_dim*mlp_ratio))
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, features):
        B = x.shape[0]
        
        _x = self.norm1(x)
        _q = self.q(_x)
        _k, _v = self.kv(features).chunk(2, dim=-1)
        _q = _q.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        _k = _k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        _v = _v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        _x = F.scaled_dot_product_attention(_q, _k, _v)
        _x = _x.transpose(1, 2).reshape(B, -1, self.embed_dim)
        x = x + _x
        
        _x = self.norm2(x)
        _x = self.mlp(_x)
        x = x + _x
        
        return x


class DERFormer(nn.Module):
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
        num_classes=7,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.patch_embed = nn.Conv2d(3, embed_dim, 16, 16)
        self.patch_embed.load_state_dict(torch.load('pretrained_weights/pe.pt'))
        
        self.extractor = IResNet(num_layers=100)
        self.extractor.load_weight('pretrained_weights/ir101.pt')
        self.extractor.keep_stages()
        
        self.proj_emb = nn.Conv2d(256, self.embed_dim, 3, 2, 1)
        
        self.proj_rep = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, self.embed_dim, 3, 2, 1)
        )
        
        self.embedding_enhancer = Enhancer(embed_dim, num_heads, mlp_ratio)
        
        self.backbone = ViT(
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            init_values,
            reg_tokens,
            pos_drop_rate,
            patch_drop_rate,
            drop_path_rate,
            num_patches=49,
        )
        self.backbone.load_weight('pretrained_weights/vit.pt')
        
        self.representation_enhancer = Enhancer(embed_dim, num_heads, mlp_ratio)
        
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate) if head_drop_rate > 0. else nn.Identity(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.emb_projector = nn.Bilinear(embed_dim, embed_dim, embed_dim)
        self.rep_projector = nn.Bilinear(embed_dim, embed_dim, embed_dim)
    
    def forward(self, x):
        B = x.shape[0]
        x = normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        f_rep, f_emb = self.extractor(x)
        
        f_emb = self.proj_emb(f_emb)
        f_emb = f_emb.reshape(B, self.embed_dim, -1).transpose(1, 2)
        
        f_rep = self.proj_rep(f_rep)
        f_rep = f_rep.reshape(B, self.embed_dim, -1).transpose(1, 2)
        
        x = self.patch_embed(x)
        x = x.reshape(B, self.embed_dim, -1).transpose(1, 2)
        
        # x = self.embedding_enhancer(x, f_emb)
        x = self.emb_projector(x, f_emb)
        
        x = self.backbone(x)
        
        # x = self.representation_enhancer(x, f_rep)
        
        x = self.rep_projector(x, f_rep)
        
        x = x.mean(dim=1)
        x = self.head(x)
        return x
