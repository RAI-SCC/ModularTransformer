This documentation is ordered based on submodules, which reflect the model structure.

This package uses automatic string parsing. In many places activation functions or other modules can be replaced with
their name as string. If the named class is defined in the standard location for the object class (e.g. torch.nn for
activation functions or qkv_maps for `Qmap`s) an instance of the class will be automatically created with the default
arguments.

## Transformer

### layers

#### attention_modules

##### attention_mechanisms

##### head_reductions
`HeadReduction`s merge the heads of multihead attention. For almost all architectures this will be `ConcatHeads`, but
this also allows implementing layers with head interaction.
This class is a thin wrapper around torch.nn.Module defining a consistent interface. Each derived class should be
initialized with the arguments:

    attention_dimension int: size of the input feature dimension
    nhead int: number of attention heads and size of the input head dimension

    device Optional[torch.device]: computation device the module is initialized on
    dtype Optional[torch.dtype]: data type of the module
    
    **kwargs: Any number of class specific keyword arguments
The `super.__init__` call should be done last, in the `__init__` of child classes.
Each child class needs to implement `attention_output_features`, which provides the number of output_features for
consistency checks.

**Currently available classes**

    ConcatHeads: collapses the head dimension by concatenating all features, default approach for most attention architectures

      Additional arguments:
        None    


##### output_modules
`OutputModule`s can be any torch.nn.Module that takes one input Tensor and provides an output Tensor of the same
shape except (possibly) in the last dimension.
This class is a thin wrapper around torch.nn.Module defining a consistent interface. Each derived class should be 
initialized with the arguments:

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
 - [ ] ***Documentation***
   - [ ] **layers**
     - [ ] **attention_modules**
       - [ ] **attention_mechanisms**
           - [ ] base.py
           - [ ] dot_product.py
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
       - [ ] base.py
       - [ ] classical.py
     - [ ] base.py
     - [ ] classical.py
   - [ ] README.md
   - [ ] Transformer base.py
   - [ ] Transformer classical.py



 - [ ] ***Tests***
   - [ ] **layers**
     - [ ] **attention_modules**
       - [ ] **attention_mechanisms**
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
