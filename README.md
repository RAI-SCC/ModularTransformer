This documentation is ordered based on submodules, which reflect the model structure.

This package uses automatic string parsing. In many places activation functions or other modules can be replaced with
their name as string. If the named class is defined in the standard location for the object class (e.g. torch.nn for
activation functions or qkv_maps for `Qmap`s) an instance of the class will be automatically created with the default
arguments.

**Note on shapes**

This package assumes Tensors that are inputs for `forward` methods to conform to the shape (H, B, S, F).
B is the batch dimension, which can be replaced by an arbitrary number of dimensions to represent other properties
(e.g. sampling (plating)) in addition to batching.
Typically, the head dimension H can be included in this, but where it occurs and is relevant (i.e. in `HeadReduction`)
it is always the first dimension.
`SelfAttentionModule` and `CrossAttentionModule`, which create the separate heads, handle the correct creation.

The sequence dimension S is only relevant for `masking` as it determines the shape of the attention matrix.
Current modules are only compatible with one sequence dimension, but custom attention mechanisms could change this.

The feature dimension F is an important input parameter for most modules.
The current setup does not support multiple feature dimensions.


## Transformer

The top level template classes of this package provide an encoder-decoder structure as used in Transformers (`Transformer`)
and RNNs (`ParallelTransformer`), which can be filled to create various architectures.
Some prebuild Transformer architectures are also provided.

**Template classes**

These combine the separate components into the appropriate module and also provide the forward function und perform
consistency checks. The components all come from submodules and their syntax and purpose is described below.

    Transformer
      Arguments:
        encoder_layer TransformerEncoderLayer: the encoder layer
        decoder_layer TransformerDecoderLayer: the decoder layer
        output_module OutputModule: output module to apply to the final decoder output
        num_encoder_layers int: number of encoder layers
        num_decoder_layers int: number of decoder layers

    ParallelTransformer
      Arguments:
        encoder_layer TransformerEncoderLayer: the encoder layer
        decoder_layer TransformerDecoderLayer: the decoder layer
        output_module OutputModule: output module to apply to the final decoder output
        num_layers int: number of layers

**Other available classes**

The main available class is `ClassicalTransformer`, which provides a very similar interface to th torch.nn.Transformer,
but slightly more customization options.

    ClassicalTransformer
      Arguments:
        input_features int: size of the input feature dimension
        d_model int: internal number of features for the attention mechanism
        nhead int: number of attention heads
        dim_feedforward: dimension of the `DoubleLinearOutputModule`s (feedforward layers) in each layer and on
         final output
        num_encoder_layers int: number of encoder layers
        num_decoder_layers int: number of decoder layers
        hidden_features int: size of the encoder output feature dimension (default: input_features)
        output_features int: size of the decoder output feature dimension (default: input_features)
        inter_activation Optional[str or Callable[[Tensor], Tensor]]: activation of the `DoubleLinearOutputModule`s
        each layer (default: ReLU())
        final_activation Optional[str or Callable[[Tensor], Tensor]]: activation of the `DoubleLinearOutputModule`
        the final output layer (default: ReLU())
        encoder_mask Optional[AttentionMatrixMask or str]: mask for encoder attention (default: None)
        decoder_mask Optional[AttentionMatrixMask or str]: mask for decoder attention (default: None)
        bias bool: If set to False, all Linear layers will not learn an additive bias (default: True)
        layer_norm bool: if False not layer norm will be applied after attention and output module (default: True)
        dropout float: dropout rate applied on the output of attention and output module (default: 0.)
        device Optional[torch.device]: computation device the module is initialized on
        dtype Optional[torch.dtype]: data type of the module


### layers

This module defines the template for encoder and decoder layers. It contains the classes `TransformerEncoderLayer` and 
`TransformerDecoderLayer`, which can be used to create them from components als well as prebuilt classes for specific variants.

It optionally adds the residual connections, layer norm and dropout to each components

**Template classes**

These combine the separate components into the appropriate module and also provide the forward function und perform
consistency checks. The components all come from submodules and their syntax and purpose is described below.

    TransformerEncoderLayer
      Arguments:
        self_attention_layer SelfAttentionModule: the self-attention block
        output_layer OutputModule: the output or feedforward layer
        residual_connection bool: If False there are no residual connections around attention block and output module (default: True)
        layer_norm bool: if False, no layer norm is applied after each sublayer (default: True)
        dropout float: dropout rate on the output of each sublayer (default: 0.)
        device Optional[torch.device]: computation device the module is initialized on
        dtype Optional[torch.dtype]: data type of the module

    TransformerDecoderLayer
      Arguments:
        self_attention_layer SelfAttentionModule: the self-attention block
        cross_attention_layer CrossAttentionModule: the cross-attention block
        output_layer OutputModule: the output or feedforward layer
        residual_connection bool: If False there are no residual connections around attention block and output module (default: True)
        layer_norm bool: if False, no layer norm is applied after each sublayer (default: True)
        dropout float: dropout rate on the output of each sublayer (default: 0.)
        device Optional[torch.device]: computation device the module is initialized on
        dtype Optional[torch.dtype]: data type of the module

**Other available classes**

These implement specific variants of self- and cross-attention. Typically, each variant implements a `EncoderLayer` and 
a `DecoderLayer` with the same prefix (e.g. `ClassicalTransformerEncoderLayer` and `ClassicalTransformerDecoderLayer`).
Classes are listed by prefix, and unless otherwise specified both accept the same arguments except that
`EncoderLayer`s never require `other_features`.

    ClassicalTransformer:
      Arguments:
        input_features int: size of the (first) input feature dimension
        other_features int: size of the other (second) input feature dimension
        d_model int: internal number of features for the attention mechanism
        nhead int: number of attention heads
        dim_feedforward int: size of the hidden layer of the `DoubleLinearOutputModule` (feedforward layer)
        output_features int: size of the output feature dimension
        mask Optional[AttentionMatrixMask or str]: mask for masked attention (default: None)
        bias bool: If set to False, all Linear layers will not learn an additive bias (default: True)
        layer_norm bool: if False not layernorm will be apllied after attention and output module (default: True)
        dropout float: dropout rate applied on the output of attention and output module (default: 0.)
        activation Optional[str or Callable[[Tensor], Tensor]]: activation of the `DoubleLinearOutputModule` (default: ReLU())
        device Optional[torch.device]: computation device the module is initialized on
        dtype Optional[torch.dtype]: data type of the module


#### attention_modules

This module defines the template for self- and cross-attention blocks. It contains the classes `SelfAttentionModule` and 
`CrossAttentionModule`, which can be used to create them from components als well as prebuilt classes for specific variants.

**Template classes**

These combine the separate components into the appropriate module and also provide the forward function und perform
consistency checks. The components all come from submodules and their syntax and purpose is described below.

    SelfAttentionModule
      Arguments:
        qkv_mapping QKVmap: mapping from input to query, key, and value
        attention_mechanism AttentionModule: performs attention with query, key, and value
        head_reduction HeadReduction: recombines the results of the heads
        output_module OutputModule: maps the recombined output to the output dimension
        nhead int: number of identical heads to create

    CrossAttentionModule
      Arguments:
        q_mapping Qmap: mapping from input_ to query
        kv_mapping KVmap: mapping from other to key and value
        attention_mechanism AttentionModule: performs attention with query, key, and value
        head_reduction HeadReduction: recombines the results of the heads
        output_module OutputModule: maps the recombined output to the output dimension
        nhead int: number of identical heads to create

**Other available classes**

These implement specific variants of self- and cross-attention. Typically, each variant implements a
`SelfAttentionModule` and a `CrossAttentionModule` with the same prefix (e.g. `ClassicalSelfAttentionModule` and
`ClassicalCrossAttentionModule`).
Classes are listed by prefix, and unless otherwise specified both accept the same arguments except that
`SelfAttentionModule`s never require `other_features`.

    Classical: attention modules as used in the classical Transformer (Vaswani et al 17)
    
      Arguments:
        input_features int: size of the (first) input feature dimension
        other_features int: size of the other (second) input feature dimension
        d_model int: internal number of features for the attention mechanism
        nhead int: number of attention heads
        output_features int: size of the output feature dimension
        mask Optional[Union[AttentionMatrixMask, str]]: mask for masked attention (default: None)
        bias bool: If set to False, the DoubleLinearOutputModule will not learn an additive bias (default: True)
        device Optional[torch.device]: computation device the module is initialized on
        dtype Optional[torch.dtype]: data type of the module


##### attention_mechanisms

`AttentionModule`s represent the attention mechanism of a Transformer and should be head independent.
This class is a thin wrapper around torch.nn.Module defining a consistent interface.
Most importantly, the `forward` method requires three inputs (query, key, and value) and provides one output.
Each derived class should be initialized with the arguments:

    q_features int: number of features of the q-component (i.e. first) input
    k_features Optional[int]: number of features of the k-component (i.e. second) input (default: q_features)
    v_features Optional[int]: number of features of the v-component (i.e. third) input (default: k_features)

    device Optional[torch.device]: computation device the module is initialized on
    dtype Optional[torch.dtype]: data type of the module
    
    **kwargs: Any number of class specific keyword arguments

The `super.__init__` call should be done last, in the `__init__` of child classes.
Each child class needs to implement `attention_output_features`, which provides the number of output_features for
consistency checks.

**Currently available classes**

    DotProductAttention: classical dot product attention mechanism (Vaswani et al 17) with optional mask

      Additional arguments:
        mask Optional[AttentionMatrixMask or str]: mask for masked attention (default: None)
      Since q_features == v_features for this mechanism, v_features is ignored and inferred.


###### masking

This module contains mask classes. Since various `AttentionModule`s might have different, incompatible ways of masking
there might be multiple class groups.
The main masking approach applies to `AttentionModule`s that create an attention matrix, where certain values are masked.
These masks are represented by `AttentionMatrixMask`s as base class and only share and require the `apply_to` method,
which accepts the attention matrix and returns a masked version of it.

The most common mask is `TriangularMask`, which limits the n-th output step of a sequence to consider only the first n
steps of the input sequence.

**Currently available classes**

    TriangularMask: works as the standard mask blocking 'future' information

      Additional arguments:
        None


##### head_reductions

`HeadReduction`s merge the heads of multihead attention. For almost all architectures this will be `ConcatHeads`, but
this also allows implementing layers with head interaction.
This class is a thin wrapper around torch.nn.Module defining a consistent interface.
Each derived class should be initialized with the arguments:

    attention_dimension int: size of the input feature dimension
    nhead int: number of attention heads and size of the input head dimension

    device Optional[torch.device]: computation device the module is initialized on
    dtype Optional[torch.dtype]: data type of the module
    
    **kwargs: Any number of class specific keyword arguments

The `super.__init__` call should be done last, in the `__init__` of child classes.
Each child class needs to implement `attention_output_features`, which provides the number of output_features for
consistency checks.

**Currently available classes**

    ConcatHeads: collapses the head dimension by concatenating all features, default approach for most attention
      architectures using a sequence of input vectors

      Additional arguments:
        None    


##### output_modules

`OutputModule`s can be any torch.nn.Module that takes one input Tensor and provides an output Tensor of the same
shape except (possibly) in the last dimension.
This class is a thin wrapper around torch.nn.Module defining a consistent interface.
Each derived class should be initialized with the arguments:

    attention_output_features int: number of input nodes and size of the feature dimension of the intended input
    output_features int: number of output features (default: attention_output_features)

    device Optional[torch.device]: computation device the module is initialized on
    dtype Optional[torch.dtype]: data type of the module

    **kwargs: Any number of class specific keyword arguments

The `super.__init__` call should be done last, in the `__init__` of child classes.

**Currently available classes**

    LinearOutputModule: a simple single layer output module with optional activation, commonly used to reduce the
      nhead * dmodel output features of classical multihead attention back to dmodel

      Additional arguments:
        activation Optional[Module or str]: output activation function (default: None)
        bias bool: If set to False, the layer will not learn an additive bias (default: True)


    DoubleLinearOutputModule: A two layer output module with optional activation after the first layer, commonly used as
      "feedforward layer" in the classical Transformer architecture.

      Additional arguments:
        dim_feedforward int: dimension of the hidden layer (default: 1024)
        activation Optional[Module or str]: intermediate activation function (default: ReLu())
        bias bool: If set to False, the layer will not learn an additive bias (default: True)


    NoModule: used when no OutputModule is required, simply assert matching dimension during initialization and forwards
      the input

      Additional arguments:
        None


##### qkv_maps

The purpose of `qkv_maps` is taking an input and producing query (*q*), key (*k*), and value (*v*) for the attention
mechanism from them.
Since this works slightly differently for cross- and self-attention there are a total of three (related) classes to
accomplish this.
`QKVmap` derives *q*, *k*, and *v* from a single input for self-attention.
`Qmap` and `KVmap` fulfill a similar purpose for cross-attention, where *q* is derived from one input and *k* and *v*
are derived from another.

Of course, there is no fundamental difference between mapping from an input to *q*, *k* or *v* and therefore `Qmap` can 
be considered the fundamental operation, since `KVmap`s and `QKVmap`s can be constructed from two or three `Qmap`s
respectively.
This is taken into account by enabling adding two `Qmap`s to obtain a `KVmap` or equivalently combining them in a
`CompositeKVmap`.
Furthermore, a `Qmap` and a `KVmap` can be added to obtain a `QKVmap` or combined in a `CompositeQKVmap`.\
Note, that the commuted version, adding a `KVmap` to a `Qmap` is intentionally disabled.
This is because adding `Qmap` + `Qmap` + `Qmap` it is intuitive that the three are q_map, k_map, v_map in that order.
However, since the first addition is performed first, it becomes `KVmap` + `Qmap` and therefore the actual order would
be k_map, v_map, q_map.
If needed these operations can be easily replaced by `Qmap` + (`Qmap` + `Qmap`) and `Qmap` + `KVmap`.\
While less convenient it is typically slightly more computationally efficient to define dedicated `KVmap`s and
`QKVmap`s

These classes are a thin wrapper around torch.Module defining a consistent interface.
Each derived class should be initialized with the arguments:

    input_features int: number of input nodes and size of the feature dimension of the intended input

    q_features int: number of output features of the q-component output
    k_features int: number of output features of the k-component output (default: q_features)
    v_features int: number of output features of the v-component output (default: k_features)

    device Optional[torch.device]: computation device the module is initialized on
    dtype Optional[torch.dtype]: data type of the module

    **kwargs: Any number of class specific keyword arguments

`Qmap`s do not require `k_features` and `v_features` and `KVmap`s do not require `q_features`. Since these features are
typically identical `v_features` defaults to the value of `k_features`, which for the `QKVmap` defaults to `q_features`.
The `super.__init__` call should be done last, in the `__init__` of child classes.

**Currently available classes:**

Unless otherwise specified all classes are available as `Qmap`, `KVmap`, and `QKVmap` with the naming convention 
[name]Q/KVmap (e.g. `LinearQmap`, `LinearKVmap`, and `LinearQKVmap`)

    Linear: single torch.nn.Linear layer mapping from input to output

      Additional arguments:
        activation Optional[Module or str]: output activation function (default: None)
        bias bool: If set to False, the layer will not learn an additive bias (default: True)
  

## ToDo:
 - [x] ***Documentation***
   - [x] **layers**
     - [x] **attention_modules**
       - [x] **attention_mechanisms**
         - [x] **masking**
           - [x] base.py
           - [x] triangular.py
         - [x] base.py
         - [x] dot_product.py
       - [x] **head_reductions**
           - [x] base.py
           - [x] concat.py
       - [x] **output_modules**
           - [x] base.py
           - [x] linear.py
           - [x] none.py
       - [x] **qkv_maps**
           - [x] base.py
           - [x] linear.py
       - [x] base.py
       - [x] classical.py
     - [x] base.py
     - [x] classical.py
   - [x] README.md
   - [x] Transformer base.py
   - [x] Transformer classical.py



 - [ ] ***Tests***
   - [ ] **layers**
     - [ ] **attention_modules**
       - [ ] **attention_mechanisms**
         - [ ] **masking**
           - [ ] base.py
           - [ ] triangular.py
           - [ ] base.py
           - [ ] dot_product.py
       - [ ] **head_reductions**
           - [ ] base.py
           - [ ] concat.py
       - [ ] **output_modules**
           - [ ] base.py
           - [ ] linear.py
           - [ ] none.py
       - [ ] **qkv_maps**
           - [ ] base.py
           - [ ] linear.py
       - [ ] base.py
       - [ ] classical.py
     - [ ] base.py
     - [ ] classical.py
   - [ ] README.md
   - [ ] Transformer base.py
   - [ ] Transformer classical.py
