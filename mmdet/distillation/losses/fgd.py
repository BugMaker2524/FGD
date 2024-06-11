import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


@DISTILL_LOSSES.register_module()
class FeatureLoss(nn.Module):
    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 temp=0.5,
                 alpha_fgd=0.5,
                 beta_fgd=0.1,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))
        self.ema_atten_t = EMA(teacher_channels)
        self.ema_atten_s = EMA(student_channels)

        self.reset_parameters()

    # def forward(self,
    #             preds_S,
    #             preds_T,
    #             gt_bboxes,
    #             img_metas):
    #     """Forward function.
    #     Args:
    #         preds_S(Tensor): Bs*C*H*W, student's feature map
    #         preds_T(Tensor): Bs*C*H*W, teacher's feature map
    #         gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #         image size, scaling factor, etc.
    #     """
    #     assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'
    #     if self.align is not None:
    #         preds_S = self.align(preds_S)
    #
    #     # [8, 256, 7, 10]
    #     N, C, H, W = preds_S.shape
    #
    #     # [B, H, W], [B, C]
    #     S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
    #     S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)
    #
    #     # [B, H, W], [B, H, W]
    #     Mask_fg = torch.zeros_like(S_attention_t)
    #     Mask_bg = torch.ones_like(S_attention_t)
    #     wmin, wmax, hmin, hmax = [], [], [], []
    #     for i in range(N):
    #         new_boxxes = torch.ones_like(gt_bboxes[i])
    #         new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W
    #         new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W
    #         new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H
    #         new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H
    #
    #         wmin.append(torch.floor(new_boxxes[:, 0]).int())
    #         wmax.append(torch.ceil(new_boxxes[:, 2]).int())
    #         hmin.append(torch.floor(new_boxxes[:, 1]).int())
    #         hmax.append(torch.ceil(new_boxxes[:, 3]).int())
    #
    #         area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
    #                 wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))
    #
    #         for j in range(len(gt_bboxes[i])):
    #             Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
    #                 torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])
    #
    #         Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)
    #         if torch.sum(Mask_bg[i]):
    #             Mask_bg[i] /= torch.sum(Mask_bg[i])
    #
    #     fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg,
    #                                          C_attention_s, C_attention_t, S_attention_s, S_attention_t)
    #     mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
    #     rela_loss = self.get_rela_loss(preds_S, preds_T)
    #
    #     loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
    #            + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
    #
    #     return loss

    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'
        if self.align is not None:
            preds_S = self.align(preds_S)

        # [8, 256, 7, 10]
        N, C, H, W = preds_S.shape

        # [B, H, W], [B, C]
        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        # [B, H, W], [B, H, W]
        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                    wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                    torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

        feat_t = self.ema_atten_t(preds_T)
        feat_s = self.ema_atten_s(preds_S)
        fg_loss, bg_loss = self.get_fbi_loss(feat_t, feat_s, Mask_fg, Mask_bg)
        loss_none = torch.tensor(0.0, device='cuda:0', requires_grad=False)
        loss_fg = self.alpha_fgd * fg_loss
        loss_bg = self.beta_fgd * bg_loss
        loss_all = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss

        return loss_all

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape

        # [B, C, H, W]
        value = torch.abs(preds)
        # [B, 1, H, W]
        fea_map = value.mean(axis=1, keepdim=True)
        # [B, H, W]
        S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

        # [B, C]
        channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        # [B, C]
        C_attention = C * F.softmax(channel_map / temp, dim=1)

        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')

        # preds_S = preds_T = [B, C, H, W]
        # [B, H, W], [B, H, W] => [B, 1, H, W], [B, 1, H, W]
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        # [B, C] => [B, C, 1] => [B, C, 1, 1]
        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        # [B, H, W] => [B, 1, H, W]
        S_t = S_t.unsqueeze(dim=1)

        # fea_t = fg_fea_t = bg_fea_t = [B, C, H, W]
        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        # fea_s = fg_fea_s = bg_fea_s = [B, C, H, W]
        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        # C_s = C_t = [B, C]
        # S_s = S_t = [B, H, W]
        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s) + torch.sum(torch.abs((S_s - S_t))) / len(S_s)

        return mask_loss

    def spatial_pool(self, x, in_type):
        # not "height, width" ???
        batch, channel, width, height = x.size()
        # [B, C, H, W]
        input_x = x
        # [B, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [B, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [B, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [B, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [B, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [B, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [B, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [B, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        # context_s = context_t = [B, C, 1, 1]
        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        # out_s = out_t = [B, C, H, W]
        out_s = preds_S
        out_t = preds_T

        # [B, C, 1, 1]
        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        # [B, C, 1, 1]
        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

        return rela_loss

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)

    def get_fbi_loss(self, fea_t, fea_s, Mask_fg, Mask_bg):
        loss_mse = nn.MSELoss(reduction='sum')
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss
