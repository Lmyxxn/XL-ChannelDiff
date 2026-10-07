import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.channels = channels+1

        self.down_blocks = nn.ModuleList([
            # 32x32 -> 16x16
            nn.Sequential(
                nn.Conv2d(self.channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2)
            ),
            # 16x16 -> 8x8
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            ),
            # 8x8 -> 4x4
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            )
        ])

        self.final = nn.Sequential(
            nn.Conv2d(256, 1, 4, 1, 0)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)

        for down_block in self.down_blocks:
            x = down_block(x)

        out = self.final(x)
        return out.view(-1, 1)

