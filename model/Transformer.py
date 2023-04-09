import math
from einops import rearrange
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer, ConvModule
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import constant_init, normal_init,trunc_normal_init
from mmcv.runner import BaseModule, ModuleList, Sequential
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from mmseg.ops import resize
from model.MaskMultiheadAttention import MaskMultiHeadAttention


class MixFFN(BaseModule):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MaskMultiHeadAttention(
            in_features=embed_dims, head_num=num_heads, bias=False, activation=None
        )
        torch.nn.MultiheadAttention

    def forward(self, x, hw_shape, source=None, identity=None, mask=None, cross=False):
        x_q = x
        if source is None:
            x_kv = x
        else:
            x_kv = source
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x_kv, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)

        if identity is None:
            identity = x_q

        out, weight = self.attn(q=x_q, k=x_kv, v=x_kv, mask=mask, cross=cross)
        return identity + self.dropout_layer(self.proj_drop(out)), weight


class TransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, hw_shape, source=None, mask=None, cross=False):
        if source is None:
            x, weight = self.attn(self.norm1(x), hw_shape, identity=x)
        else:
            x, weight = self.attn(self.norm1(x), hw_shape, source=self.norm1(source), identity=x, mask=mask, cross=cross)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x, weight


class MixVisionTransformer(BaseModule):
    def __init__(self,
                 shot=1,
                 in_channels=64,
                 num_similarity_channels = 2,
                 num_down_stages = 3,
                 embed_dims = 64,
                 num_heads = [2, 4, 8],
                 match_dims = 64, 
                 match_nums_heads = 2,
                 down_patch_sizes = [1, 3, 3],
                 down_stridess = [1, 2, 2],
                 down_sr_ratio = [4, 2, 1],
                 mlp_ratio=4,
                 drop_rate=0.1,
                 attn_drop_rate=0.,
                 qkv_bias=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super(MixVisionTransformer, self).__init__(init_cfg=init_cfg)
        self.shot = shot

        #-------------------------------------------------------- Self Attention for Down Sample ------------------------------------------------------------
        self.num_similarity_channels = num_similarity_channels
        self.num_down_stages = num_down_stages
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.match_dims = match_dims
        self.match_nums_heads = match_nums_heads
        self.down_patch_sizes = down_patch_sizes
        self.down_stridess = down_stridess
        self.down_sr_ratio = down_sr_ratio
        self.mlp_ratio=mlp_ratio
        self.qkv_bias = qkv_bias
        self.down_sample_layers = ModuleList()
        for i in range(num_down_stages):
            self.down_sample_layers.append(nn.ModuleList([
                PatchEmbed(
                    in_channels=embed_dims,
                    embed_dims=embed_dims,
                    kernel_size=down_patch_sizes[i],
                    stride=down_stridess[i],
                    padding=down_stridess[i] // 2,
                    norm_cfg=norm_cfg),
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=down_sr_ratio[i]),
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=down_sr_ratio[i]),
                build_norm_layer(norm_cfg, embed_dims)[1]
            ]))

        #-------------------------------------------------------- Corss Attention for Down Matching ------------------------------------------------------------
        self.match_layers = ModuleList()
        for i in range(self.num_down_stages):
            level_match_layers = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=self.match_dims,
                    num_heads=self.match_nums_heads,
                    feedforward_channels=self.mlp_ratio * self.match_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=1
                ),
                ConvModule(self.match_dims + self.num_similarity_channels, self.match_dims, kernel_size=3, stride=1, padding=1, norm_cfg=dict(type="SyncBN"))])
            self.match_layers.append(level_match_layers)
        
        self.parse_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims),
            nn.ReLU()
        ) for _ in range(self.num_down_stages)
        ])

        self.cls = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, 2, kernel_size=1, stride=1, padding=0)
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(MixVisionTransformer, self).init_weights()

    def forward(self, q_x, s_x, mask, similarity):
        down_query_features = []
        down_support_features = []
        hw_shapes = []
        down_masks = []
        down_similarity = []
        weights = []
        for i, layer in enumerate(self.down_sample_layers):
            q_x, q_hw_shape = layer[0](q_x)
            s_x, s_hw_shape = layer[0](s_x)
            q_x, s_x = layer[1](q_x, hw_shape=q_hw_shape)[0], layer[1](s_x, hw_shape=s_hw_shape)[0]
            q_x, s_x = layer[2](q_x, hw_shape=q_hw_shape)[0], layer[2](s_x, hw_shape=s_hw_shape)[0]
            q_x, s_x = layer[3](q_x), layer[3](s_x)
            tmp_mask = resize(mask, s_hw_shape, mode="nearest")  
            tmp_mask = rearrange(tmp_mask, "(b n) 1 h w -> b 1 (n h w)", n=self.shot)
            tmp_mask = tmp_mask.repeat(1, q_hw_shape[0] * q_hw_shape[1], 1)
            tmp_similarity = resize(similarity, q_hw_shape, mode="bilinear", align_corners=True)
            down_query_features.append(q_x)
            down_support_features.append(rearrange(s_x, "(b n) l c -> b (n l) c", n=self.shot))
            hw_shapes.append(q_hw_shape)
            down_masks.append(tmp_mask)
            down_similarity.append(tmp_similarity)
            if i != self.num_down_stages - 1:
                q_x, s_x = nlc_to_nchw(q_x, q_hw_shape), nlc_to_nchw(s_x, s_hw_shape)

        outs = None
        for i in range(self.num_down_stages).__reversed__():
            layer = self.match_layers[i]
            out, weight = layer[0](
                x=down_query_features[i], 
                hw_shape=hw_shapes[i], 
                source=down_support_features[i], 
                mask=down_masks[i], 
                cross=True)
            out = nlc_to_nchw(out, hw_shapes[i])
            weight = weight.view(out.shape[0], hw_shapes[i][0], hw_shapes[i][1])
            out = layer[1](torch.cat([out, down_similarity[i]], dim=1))
            weights.append(weight)
            # print(layer_out.shape) 
            if outs is None:
                outs = self.parse_layers[i](out)
            else:
                outs = resize(outs, size=out.shape[-2:], mode="bilinear")
                outs = outs + self.parse_layers[i](out + outs)
        outs = self.cls(outs)
        return outs, weights


class Transformer(nn.Module):
    def __init__(self, shot=1) -> None:
        super().__init__()
        self.shot=shot
        self.mix_transformer = MixVisionTransformer(shot=self.shot)
  
    def forward(self, features, supp_features, mask, similaryty):
        shape = features.shape[-2:]
        outs, weights = self.mix_transformer(features, supp_features, mask, similaryty)
        return outs, weights
