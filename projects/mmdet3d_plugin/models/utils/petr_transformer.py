# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
import copy
import torch.utils.checkpoint as cp

from typing import Optional, Tuple
Tensor = torch.Tensor

def _split_scaled_dot_product_attention(
        query_split: list,
        key_split: list,
        value_split: list,
        num_heads: int, 
        num_split: int, 
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:

    tgt_len, bsz, embed_dim = query_split[0].size()
    assert num_split * (num_heads // num_split) == num_heads, f"num_heads {num_heads} not divisible by num_split {num_split}"
    head_dim = embed_dim // (num_heads // num_split)
    assert head_dim * (num_heads // num_split) == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads//num_splits {num_heads//num_split}"

    if attn_mask is not None:
        attn_mask_split = attn_mask.chunk(num_split)
    attn = []
    output = []
    for i in range(num_split):
        query_split[i] = query_split[i].view(tgt_len, bsz * num_heads // num_split , head_dim).transpose(0, 1)
        query_split[i] = query_split[i] / math.sqrt(query_split[i].shape[-1])
        key_split[i]   = key_split[i].view(-1, bsz * num_heads // num_split, head_dim).transpose(0, 1)
        value_split[i] = value_split[i].view(-1, bsz * num_heads // num_split, head_dim).transpose(0, 1)
        attn_split = torch.bmm(query_split[i], key_split[i].transpose(-2, -1))
        if attn_mask is not None:
            attn_split += attn_mask_split[i]
        attn_split = F.softmax(attn_split, dim=-1)
        if dropout_p > 0.0:
            attn_split = F.dropout(attn_split, p=dropout_p)
        output_split = torch.bmm(attn_split, value_split[i])
        attn.append(attn_split)
        output.append(output_split)

    attn_output = torch.cat(output, dim=0)
    attn_output_weights = torch.cat(attn, dim=0)

    return attn_output, attn_output_weights

class SplitPETRMultiheadAttention(nn.MultiheadAttention):
    
    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.,
                 num_head_split=1,
                 **kwargs):
        super(SplitPETRMultiheadAttention, self).__init__(embed_dims,
                                                          num_heads,
                                                          dropout)
        self.num_split = num_head_split
    
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        num_heads = self.num_heads

        in_proj_weight = self.state_dict()['in_proj_weight']
        in_proj_bias = self.state_dict()['in_proj_bias']
        out_proj_weight = self.state_dict()['out_proj.weight']
        out_proj_bias = self.state_dict()['out_proj.bias']

        
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
        
        # input projection
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        q_split = list(q.chunk(self.num_split, dim=-1))
        k_split = list(k.chunk(self.num_split, dim=-1))
        v_split = list(v.chunk(self.num_split, dim=-1))

        # update source sequence length after adjustments
        src_len = k.view(-1, bsz * num_heads, head_dim).transpose(0, 1).size(1)
        
        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
        
        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        
        # adjust dropout probability
        if not self.training:
            self.dropout = 0.0

        attn_output, attn_output_weights = _split_scaled_dot_product_attention(q_split, k_split, v_split, num_heads, self.num_split, attn_mask, self.dropout)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

@TRANSFORMER.register_module()
class PETRTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(PETRTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True


    def forward(self, x, mask, query_embed, pos_embed, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]
        target = torch.zeros_like(query_embed)

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return  out_dec, memory

@TRANSFORMER.register_module()
class PETRDNTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(PETRDNTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True


    def forward(self, x, mask, query_embed, pos_embed, attn_masks=None, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        query_embed = query_embed.transpose(0, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return  out_dec, memory

@TRANSFORMER_LAYER.register_module()
class PETRTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 with_cp=True,
                 **kwargs):
        super(PETRTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.use_checkpoint = with_cp
    
    def _forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                idx=None,
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerDecoderLayer, self).forward(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                idx=idx
                )

        return x

    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                idx=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                )
        else:
            x = self._forward(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
            idx=idx
            )
        return x

@ATTENTION.register_module()
class PETRMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 num_head_split=2,
                 split_proj=False,
                 **kwargs):
        super(PETRMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.num_head_split = num_head_split
        self.split_proj = split_proj
        if num_head_split == 1:
            self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                            **kwargs)
        else:
            if split_proj:
                self.attn_head_splits = nn.ModuleList([
                    nn.MultiheadAttention(embed_dims//self.num_head_split,
                                        num_heads//self.num_head_split,
                                        attn_drop,
                                        **kwargs) 
                                        for _ in range(2*self.num_head_split)])
            else:
                self.attn = SplitPETRMultiheadAttention(embed_dims,
                                                        num_heads,
                                                        dropout=attn_drop,
                                                        num_head_split=num_head_split)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                idx=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        if self.num_head_split >= 1:
            out = self.attn(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]
        else:
            # split multi-head (including split of input projections)
            for split in self.attn_head_splits:
                split.eval()
                if query.device.type == 'cuda':
                    split.eval().to(query.device)

            assert len(query.size()) == 3, f"Expected query of dim=3 but got {len(query.size())}"
            assert len(key.size())   == 3, f"Expected key of dim=3 but got {len(key.size())}"
            assert len(value.size()) == 3, f"Expected value of dim=3 but got {len(value.size())}"

            # attn_sd = self.attn.state_dict()
            name = f"petr_mha_{idx}.pth"
            attn_sd = torch.load(name, map_location=query.device)
            print(f"{name} loaded")

            w_q, w_k, w_v = attn_sd['in_proj_weight'].chunk(3)
            b_q, b_k, b_v = attn_sd['in_proj_bias'].chunk(3)
            w_out = attn_sd['out_proj.weight']
            b_out = attn_sd['out_proj.bias']
            b_out_split = torch.zeros(self.embed_dims//self.num_head_split)
            w_out_split = torch.eye(self.embed_dims//self.num_head_split)

            outs_splits = []
            for i in range(self.num_head_split):
                query_split = query[:, :, i*self.embed_dims//self.num_head_split:(i+1)*self.embed_dims//self.num_head_split]
                key_split   = key  [:, :, i*self.embed_dims//self.num_head_split:(i+1)*self.embed_dims//self.num_head_split]
                value_split = value[:, :, i*self.embed_dims//self.num_head_split:(i+1)*self.embed_dims//self.num_head_split]

                for j in range(self.num_head_split):
                    b_q_split   = b_q[j*self.embed_dims//self.num_head_split:(j+1)*self.embed_dims//self.num_head_split] / 2
                    b_k_split   = b_k[j*self.embed_dims//self.num_head_split:(j+1)*self.embed_dims//self.num_head_split] / 2
                    b_v_split   = b_v[j*self.embed_dims//self.num_head_split:(j+1)*self.embed_dims//self.num_head_split] / 2

                    attn_split_sd = self.attn_head_splits[i*self.num_head_split+j].state_dict()

                    # Input projection
                    w_q_split = w_q.transpose(0,1)[i*self.embed_dims//self.num_head_split:(i+1)*self.embed_dims//self.num_head_split,
                                                j*self.embed_dims//self.num_head_split:(j+1)*self.embed_dims//self.num_head_split].transpose(0,1)
                    w_k_split = w_k.transpose(0,1)[i*self.embed_dims//self.num_head_split:(i+1)*self.embed_dims//self.num_head_split,
                                                j*self.embed_dims//self.num_head_split:(j+1)*self.embed_dims//self.num_head_split].transpose(0,1)
                    w_v_split = w_v.transpose(0,1)[i*self.embed_dims//self.num_head_split:(i+1)*self.embed_dims//self.num_head_split,
                                                j*self.embed_dims//self.num_head_split:(j+1)*self.embed_dims//self.num_head_split].transpose(0,1)
                    attn_split_sd['in_proj_weight'] = torch.cat([w_q_split, w_k_split, w_v_split])
                    attn_split_sd['in_proj_bias']   = torch.cat([b_q_split, b_k_split, b_v_split])

                    # Output projection
                    attn_split_sd['out_proj.weight'] = w_out_split
                    attn_split_sd['out_proj.bias']   = b_out_split

                    self.attn_head_splits[i*self.num_head_split+j].load_state_dict(attn_split_sd)
                    out_split = self.attn_head_splits[i*self.num_head_split+j](query=query_split,
                                                                               key=key_split,
                                                                               value=value_split)[0]
                    outs_splits.append(out_split)
            scaled_dot_attn_splits = torch.cat([outs_splits[0]+outs_splits[2], outs_splits[1]+outs_splits[3]], dim=2)
            attn_split_output = F.linear(scaled_dot_attn_splits, w_out, b_out)
            out = attn_split_output

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.
    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(PETRTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(PETRTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for idx, layer in enumerate(self.layers):
            query = layer(query, *args, idx=idx, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)

