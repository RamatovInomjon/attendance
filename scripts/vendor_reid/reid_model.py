# -*- coding: utf-8 -*-
"""Asosiy tarmoq (backbone) + BNNeck bosh qismi."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# osnet.py va resnet_ibn_a.py deep-person-reid dan ko'chirilgan (vendor/ ichida)
from vendor.osnet import osnet_x0_25, osnet_x0_5, osnet_x0_75, osnet_x1_0
from vendor.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a


# --------------------------------------------------------------------------- #
#  Oldindan o'qitilgan vaznlarni yuklash
# --------------------------------------------------------------------------- #
#
# DIQQAT: oddiy `resnet50` ImageNet vaznlari IBN tarmog'iga TO'LIQ mos kelmaydi,
# chunki IBN bloklari BatchNorm ning yarmini InstanceNorm bilan almashtiradi.
# O'lchandi: oddiy resnet50 -> qatlamlarning 62% i mos; IBN-Net ning o'z
# ImageNet vaznlari -> 84%; FastReID ning ReID-da o'qitilgan vaznlari -> 99%.
# Oddiy vaznlar bilan tarmoqning 38% i tasodifiy qoladi va bu IBN modelini
# adolatsiz ravishda zaiflashtiradi.

IBN_IMAGENET = {
    "resnet50_ibn": "resnet50_ibn_a-d9d0bb7b.pth",
    "resnet101_ibn": "resnet101_ibn_a-59ea0ac6.pth",
}
REID_ZOO = {
    "resnet50_ibn": "market_sbs_R50-ibn.pth",
    "resnet101_ibn": "market_sbs_R101-ibn.pth",
}


def load_backbone_weights(net: nn.Module, arch: str, init: str,
                          cache: str = "", zoo: str = "") -> str:
    """`init`: imagenet | reid | none. Mos kelgan qatlamlar ulushini qaytaradi."""
    import os
    if init == "none":
        return "init=none"
    cache = cache or os.path.expanduser("~/.cache/torch/hub/checkpoints")
    zoo = zoo or os.environ.get("REID_ZOO_DIR") or \
        "/home/myid/projectAI/person_reid/zoo"
    if init == "imagenet":
        path = os.path.join(cache, IBN_IMAGENET.get(arch, ""))
        prefix = ""
    else:
        path = os.path.join(zoo, REID_ZOO.get(arch, ""))
        prefix = "backbone."
    if not os.path.isfile(path):
        return f"init={init}: FAYL YO'Q ({path})"
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    if prefix:
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    md = net.state_dict()
    ok = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
    net.load_state_dict(ok, strict=False)
    return f"init={init}: {len(ok)}/{len(md)} qatlam ({100*len(ok)/len(md):.0f}%)"


def _weights_init_kaiming(m: nn.Module) -> None:
    cn = m.__class__.__name__
    if cn.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif cn.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif cn.find("BatchNorm") != -1 and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ReIDNet(nn.Module):
    """Backbone -> GeM/GAP -> BNNeck -> (embedding, logits).

    BNNeck (Luo va b., 2019): triplet yo'qotishi BN dan OLDINGI xususiyatga,
    tasniflash esa BN dan KEYINGI xususiyatga qo'llaniladi. Inferensda
    BN dan keyingi xususiyat ishlatiladi.
    """

    # Xususiyat o'lchami QO'LDA yozilmaydi -- u sinov o'tkazish orqali aniqlanadi,
    # chunki OSNet variantlarida u sig'imga qarab o'zgaradi
    # (x1_0->512, x0_75->384, x0_5->256, x0_25->128).
    ARCH = {
        "osnet_x0_25": osnet_x0_25,
        "osnet_x0_5": osnet_x0_5,
        "osnet_x0_75": osnet_x0_75,
        "osnet_x1_0": osnet_x1_0,
        "resnet50_ibn": resnet50_ibn_a,
        "resnet101_ibn": resnet101_ibn_a,
    }

    def __init__(self, arch: str, num_classes: int, last_stride: int = 1,
                 pretrained: bool = True, embed_dim: int = 0,
                 init: str = "imagenet"):
        """embed_dim > 0 bo'lsa, barcha arxitekturalar UMUMIY o'lchamli
        embeddingga proyeksiya qilinadi. Bu sig'im o'qini embedding
        o'lchamidan ajratadi -- aks holda taqqoslash nazorat ostida bo'lmaydi
        (OSNet-x0.25 -> 128, ResNet50-IBN -> 2048)."""
        super().__init__()
        if arch not in self.ARCH:
            raise ValueError(f"noma'lum arxitektura: {arch}")
        fn = self.ARCH[arch]
        self.arch = arch

        if arch.startswith("osnet"):
            net = fn(num_classes=1000, pretrained=pretrained, loss="triplet")
            self.init_report = f"init=imagenet(osnet) pretrained={pretrained}"
            self.backbone = nn.Sequential(
                net.conv1, net.maxpool, net.conv2, net.conv3, net.conv4, net.conv5
            )
        else:
            net = fn(num_classes=1000, loss="triplet", pretrained=False)
            self.init_report = load_backbone_weights(net, arch, init)
            if last_stride == 1:      # ReID uchun oxirgi bosqich qadamini 1 ga tushirish
                net.layer4[0].conv2.stride = (1, 1)
                net.layer4[0].downsample[0].stride = (1, 1)
            self.backbone = nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3, net.layer4,
            )

        self.gap = nn.AdaptiveAvgPool2d(1)
        # chiqish o'lchamini sinov tenzori bilan aniqlaymiz
        with torch.no_grad():
            was = self.backbone.training
            self.backbone.eval()
            feat_dim = self.backbone(torch.zeros(1, 3, 256, 128)).shape[1]
            self.backbone.train(was)
        backbone_dim = int(feat_dim)
        if embed_dim and embed_dim != backbone_dim:
            self.reduce = nn.Sequential(
                nn.Linear(backbone_dim, embed_dim, bias=False),
                nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True))
            self.reduce.apply(_weights_init_kaiming)
            feat_dim = embed_dim
        else:
            self.reduce = nn.Identity()
        self.backbone_dim, self.feat_dim = backbone_dim, int(feat_dim)
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)      # BNNeck: siljishsiz
        self.bottleneck.apply(_weights_init_kaiming)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.classifier.apply(_weights_init_kaiming)

    def forward(self, x: torch.Tensor):
        f = self.reduce(self.gap(self.backbone(x)).flatten(1))  # triplet uchun (BN dan oldin)
        fb = self.bottleneck(f)                     # inferens/tasniflash uchun
        if not self.training:
            return F.normalize(fb, dim=1)
        return f, self.classifier(fb)


# --------------------------------------------------------------------------- #
#  Yo'qotish funksiyalari
# --------------------------------------------------------------------------- #

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes, self.eps = num_classes, epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = (1 - self.eps) * t + self.eps / self.num_classes
        return (-t * logp).sum(dim=1).mean()


class TripletLoss(nn.Module):
    """Qiyin namunalarni tanlash bilan (batch-hard), yumshoq margin."""

    def __init__(self, margin: float | None = None):
        super().__init__()
        self.margin = margin
        self.ranking = (nn.MarginRankingLoss(margin=margin) if margin
                        else nn.SoftMarginLoss())

    def forward(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = feat.size(0)
        d = torch.cdist(feat, feat, p=2).clamp(min=1e-12)
        mask = target.expand(n, n).eq(target.expand(n, n).t())
        dist_ap = (d * mask.float() - 1e6 * (~mask).float()).max(dim=1)[0]
        dist_an = (d * (~mask).float() + 1e6 * mask.float()).min(dim=1)[0]
        y = torch.ones_like(dist_an)
        if self.margin:
            return self.ranking(dist_an, dist_ap, y)
        return self.ranking(dist_an - dist_ap, y)


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = self.centers[target]
        return ((feat - c) ** 2).sum(dim=1).mean() / 2
