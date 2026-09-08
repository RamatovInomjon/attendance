# -*- coding: utf-8 -*-
"""Tasniflash boshlari: softmax (hozirgi), Circle, AdaFace.

Uchalasi ham BIR XIL interfeysga ega:  head(fb, f, target) -> loss

    fb  BNNeck dan KEYINGI xususiyat (inferensda ishlatiladigan)
    f   BNNeck dan OLDINGI xususiyat (triplet shu yerda ishlaydi)
    target  identifikator yorliqlari

Nima uchun ikkala xususiyat ham uzatiladi: AdaFace marginni xususiyat
NORMASIDAN oladi, BatchNorm esa normani yo'q qiladi. Yo'nalish `fb` dan
(inferens bilan mos), sifat signali `‖f‖` dan olinadi.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxHead(nn.Module):
    """Hozirgi holat: yorliq silliqlangan oddiy softmax (BoT retsepti)."""

    def __init__(self, feat_dim: int, num_classes: int, eps: float = 0.1):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.eps, self.num_classes = eps, num_classes

    def forward(self, fb, f, target):
        logp = F.log_softmax(self.classifier(fb), dim=1)
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = (1 - self.eps) * t + self.eps / self.num_classes
        return (-t * logp).sum(dim=1).mean()


class CircleHead(nn.Module):
    """Circle loss, sinf darajasidagi (softmax) shakli.

    Sun va b., CVPR 2020. FastReID `market_sbs` vaznlari aynan shu bilan
    o'qitilgan -- bizda `reid` init e512 da foyda bergani (-1.79 FNIR)
    shu retseptning bilvosita tasdig'i.

    Gap ArcFace/CosFace dan farqi: margin QAT'IY emas, har bir juftlik
    o'zining optimal nuqtasidan qanchalik uzoqligiga qarab vaznlanadi
    (alpha_p, alpha_n). Ya'ni allaqachon yaxshi ajratilgan juftliklar
    kamroq gradient oladi.
    """

    def __init__(self, feat_dim: int, num_classes: int, s: float = 64.0,
                 m: float = 0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.s, self.m, self.num_classes = s, m, num_classes

    def forward(self, fb, f, target):
        cos = F.linear(F.normalize(fb), F.normalize(self.weight)).clamp(-1, 1)
        a_p = torch.clamp_min(-cos.detach() + 1 + self.m, min=0.0)
        a_n = torch.clamp_min(cos.detach() + self.m, min=0.0)
        d_p, d_n = 1 - self.m, self.m
        s_p = self.s * a_p * (cos - d_p)
        s_n = self.s * a_n * (cos - d_n)
        oh = torch.zeros_like(cos).scatter_(1, target.unsqueeze(1), 1)
        logits = oh * s_p + (1 - oh) * s_n
        return F.cross_entropy(logits, target)


class AdaFaceHead(nn.Module):
    """AdaFace: sifatga moslashuvchi margin (Kim va b., CVPR 2022).

    Margin xususiyat NORMASIGA qarab o'zgaradi: norma tasvir sifatining
    o'rinbosari. Sifatli namunada margin KATTA (qattiqroq o'qitish),
    sifatsizida KICHIK yoki manfiy (tarmoq undan chekinadi).

    Nima uchun bizga mos: E2 tajribasi sifatsiz manbalarni O'CHIRISH
    9.8 FNIR punkt zarar keltirishini ko'rsatdi, ya'ni ularda signal bor.
    Ochiq qolgan savol -- ularni o'chirish emas, VAZNINI KAMAYTIRISH.
    AdaFace aynan shu mexanizm.

    ⚠ Norma `f` dan (BNNeck dan OLDIN) olinadi. `fb` da BatchNorm normani
    yo'q qilgan bo'lardi va margin hech narsaga moslashmasdi -- bu holda
    loss jimgina oddiy CosFace ga aylanadi.
    """

    def __init__(self, feat_dim: int, num_classes: int, m: float = 0.4,
                 h: float = 0.333, s: float = 64.0, t_alpha: float = 0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.m, self.h, self.s, self.t_alpha = m, h, s, t_alpha
        # normaning yuruvchi statistikasi: partiya kichik bo'lsa ham barqaror
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))

    def forward(self, fb, f, target):
        cos = F.linear(F.normalize(fb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        norm = torch.norm(f.detach(), 2, 1, keepdim=True).clamp(0.001, 100)
        with torch.no_grad():
            self.batch_mean.mul_(1 - self.t_alpha).add_(self.t_alpha * norm.mean())
            self.batch_std.mul_(1 - self.t_alpha).add_(self.t_alpha * norm.std().nan_to_num(1.0))
        scaler = ((norm - self.batch_mean) / (self.batch_std + 1e-3)
                  ).mul(self.h).clamp(-1, 1)                     # [-1, 1]

        oh = torch.zeros_like(cos).scatter_(1, target.unsqueeze(1), 1)
        # burchak margini: sifat past bo'lsa musbat (yengillashtiradi)
        theta = torch.acos(cos)
        theta_m = (theta + oh * (-self.m * scaler)).clamp(1e-7, math.pi - 1e-7)
        cos_m = torch.cos(theta_m)
        # qo'shimcha kosinus margini
        cos_final = cos_m - oh * (self.m + self.m * scaler)
        return F.cross_entropy(self.s * cos_final, target)


BOSHLAR = {"softmax": SoftmaxHead, "circle": CircleHead, "adaface": AdaFaceHead}
