import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa import SpecAugmentation

from .layers import ConvBlock
from ..feature_fusion import iAFF


class Cnn14(nn.Module):
    """Cnn14 model as described in https://doi.org/10.1109/TASLP.2020.3030497.
    Implementation adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn.
    """
    def __init__(self, config: dict):
        super().__init__()

        self.use_fusion = config["use_fusion"]
        emb_out = config["out_size"]
        classes_num = config["num_classes"]

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.fc1 = nn.Linear(2048, emb_out, bias=True)
        self.fc_audioset = nn.Linear(emb_out, classes_num, bias=True)

        if self.use_fusion:
            self.mel_conv2d = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(6, 2), padding=(2, 2)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.fusion_model = iAFF(channels=64, type="2D")

    def forward(self, x: dict[str, torch.Tensor]):
        longer_list = x["is_longer"]
        longer_list_idx = torch.where(longer_list)[0]

        x = x["audio"]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x_global = x[:, 0:1, :, :]

        x_global = self.conv_block1(x_global, pool_size=(2, 2), pool_type='avg')

        # Fuse samples that are longer
        if len(longer_list_idx) > 0:
            x_local = x[longer_list_idx, 1:, :, :].contiguous()
            TH = x_global.size(-2)

            # local processing
            B, C, H, W = x_local.shape
            x_local = x_local.view(B * C, 1, H, W)
            x_local = self.mel_conv2d(x_local)
            x_local = x_local.view(B, C, x_local.size(1), x_local.size(2), x_local.size(3))
            x_local = x_local.permute((0, 2, 1, 3, 4)).contiguous().flatten(2, 3)

            TB, TC, _, TW = x_local.size()
            if x_local.size(-2) < TH:
                x_local = torch.cat([x_local, torch.zeros((TB, TC, TH - x_local.size(-2), TW), device=x_global.device)],
                                    dim=-2)
            else:
                x_local = x_local[:, :, :TH, :]

            # Fuse the global and local samples
            x_global[longer_list_idx] = self.fusion_model(x_global[longer_list_idx], x_local)

        x = x_global

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clip_wise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clip_wise_output': clip_wise_output, 'embedding': embedding}

        return output_dict
