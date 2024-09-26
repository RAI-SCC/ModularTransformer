from typing import Optional, Union

import torch

from .attention_mechanisms import DotProductAttention
from .attention_mechanisms.masking import AttentionMatrixMask
from .base import CrossAttentionModule, SelfAttentionModule
from .head_reductions import ConcatHeads
from .output_modules import LinearOutputModule
from .qkv_maps import LinearMCDKVmap, LinearMCDQKVmap, LinearMCDQmap

__all__ = [
    "ClassicalMCDSelfAttentionModule",
    "ClassicalMCDCrossAttentionModule",
]


class ClassicalMCDSelfAttentionModule(SelfAttentionModule):
    """
    Self attention module as used in the classical Transformer (Vaswani et al 17)

    Parameters
        :param input_features int: size of the input feature dimension
        :param d_model int: internal number of features for the attention mechanism
        :param nhead int: number of attention heads
        :param output_features int: size of the output feature dimension
        :param mask Optional[AttentionMatrixMask or str]: mask for masked attention (default: None)
        :param bias bool: If set to False, all Linear layers will not learn an additive bias (default: True)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
        :param gaussian bool: If True use Gaussian dropout, else Bernoulli dropout
        :param weight_drop bool: If True use weight dropout, else node dropout
        :param rate float: Dropout rate Bernoulli dropout
        :param std_dev float: Standard deviation for Gaussian dropout
        :param istrainablesigma bool: If True train sigma for Gaussian weight dropout.
    """

    def __init__(
        self,
        input_features: int,
        d_model: int,
        nhead: int,
        output_features: int,
        mask: Optional[Union[AttentionMatrixMask, str]] = None,
        bias: bool = True,
        gaussian: bool = False,
        weight_drop: bool = False,
        rate: float = 0.5,
        std_dev: float = 0.5,
        istrainablesigma: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        qkv_mapping = LinearMCDQKVmap(
            input_features=input_features,
            q_features=d_model,
            k_features=d_model,
            v_features=d_model,
            activation=None,
            bias=bias,
            gaussian=gaussian,
            weight_drop=weight_drop,
            rate=rate,
            std_dev=std_dev,
            istrainablesigma=istrainablesigma,
            **factory_kwargs,
        )

        attention_mechanism = DotProductAttention(
            q_features=d_model, v_features=d_model, mask=mask, **factory_kwargs
        )

        head_reduction = ConcatHeads(attention_dimension=d_model, nhead=nhead, **factory_kwargs)
        attention_output_features = head_reduction.attention_output_features

        output_module = LinearOutputModule(
            attention_output_features=attention_output_features,
            output_features=output_features,
            activation=None,
            bias=bias,
            **factory_kwargs,
        )

        super().__init__(
            qkv_mapping=qkv_mapping,
            attention_mechanism=attention_mechanism,
            head_reduction=head_reduction,
            output_module=output_module,
            nhead=nhead,
        )


class ClassicalMCDCrossAttentionModule(CrossAttentionModule):
    """
    Cross attention module as used in the classical Transformer (Vaswani et al 17)

    Parameters
        :param input_features int: size of the (first) input feature dimension
        :param other_features int: size of the other (second) input feature dimension
        :param d_model int: internal number of features for the attention mechanism
        :param nhead int: number of attention heads
        :param output_features int: size of the output feature dimension
        :param mask Optional[AttentionMatrixMask or str]: mask for masked attention (default: None)
        :param bias bool: If set to False, all Linear layers will not learn an additive bias (default: True)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
        :param gaussian bool: If True use Gaussian dropout, else Bernoulli dropout
        :param weight_drop bool: If True use weight dropout, else node dropout
        :param rate float: Dropout rate Bernoulli dropout
        :param std_dev float: Standard deviation for Gaussian dropout
        :param istrainablesigma bool: If True train sigma for Gaussian weight dropout.
    """

    def __init__(
        self,
        input_features: int,
        other_features: int,
        d_model: int,
        nhead: int,
        output_features: int,
        mask: Optional[Union[AttentionMatrixMask, str]] = None,
        bias: bool = True,
        gaussian: bool = False,
        weight_drop: bool = False,
        rate: float = 0.5,
        std_dev: float = 0.5,
        istrainablesigma: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        q_mapping = LinearMCDQmap(
            input_features=input_features,
            q_features=d_model,
            activation=None,
            bias=bias,
            gaussian=gaussian,
            weight_drop=weight_drop,
            rate=rate,
            std_dev=std_dev,
            istrainablesigma=istrainablesigma,
            **factory_kwargs,
        )

        kv_mapping = LinearMCDKVmap(
            input_features=other_features,
            k_features=d_model,
            v_features=d_model,
            activation=None,
            bias=bias,
            gaussian=gaussian,
            weight_drop=weight_drop,
            rate=rate,
            std_dev=std_dev,
            istrainablesigma=istrainablesigma,
            **factory_kwargs,
        )

        attention_mechanism = DotProductAttention(
            q_features=d_model, v_features=d_model, mask=mask, **factory_kwargs
        )

        head_reduction = ConcatHeads(attention_dimension=d_model, nhead=nhead, **factory_kwargs)
        attention_output_features = head_reduction.attention_output_features

        output_module = LinearOutputModule(
            attention_output_features=attention_output_features,
            output_features=output_features,
            activation=None,
            bias=bias,
            **factory_kwargs,
        )

        super().__init__(
            q_mapping=q_mapping,
            kv_mapping=kv_mapping,
            attention_mechanism=attention_mechanism,
            head_reduction=head_reduction,
            output_module=output_module,
            nhead=nhead,
        )
