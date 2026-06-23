import torch
from torch import nn
from timm.models.vision_transformer import Block
from timm.layers import DropPath, PatchDropout


class ViT(nn.Module):
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
        drop_path_rate=0.1,
        num_patches=0,
    ):
        super().__init__()
        
        self.num_reg_tokens = reg_tokens
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 1e-6) if num_patches else None
        self.reg_token = nn.Parameter(torch.randn(1, reg_tokens, embed_dim) * 1e-6) if reg_tokens else None
        self.pos_drop = nn.Dropout(pos_drop_rate, inplace=True) if pos_drop_rate > 0. else nn.Identity()
        self.patch_drop = PatchDropout(
            patch_drop_rate,
            num_prefix_tokens=reg_tokens,
        ) if patch_drop_rate > 0 else nn.Identity()
                
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                init_values=init_values,
                drop_path=dpr[i],
            ) for i in range(depth)
        ])
            
    def load_weight(self, f):
        self.load_state_dict(torch.load(f))
    
    def forward(self, x):
        if self.pos_embed is not None:
            x = x + self.pos_embed
        if self.reg_token is not None:
            x = torch.cat([self.reg_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = x[:, self.num_reg_tokens:]
        return x
