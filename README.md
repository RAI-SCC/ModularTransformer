# Modular Transformer
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
## Motivation
There is a vast and quickly growing space of existing transformer archictures and attention models.
To facilitate exploring this space, experimenting with, and comparing different models Modular Transformer aims to provide a convenient interface to a zoo of models, building blocks, and layers.

## Structure

.
├── base.py
├── classical.py
├── layers
│   ├── attention_modules
│   │   ├── attention_mechanisms
│   │   │   ├── base.py
│   │   │   ├── dot_product.py
│   │   │   └── masking
│   │   │       ├── base.py
│   │   │       └── triangular.py
│   │   ├── base.py
│   │   ├── classical.py
│   │   ├── head_reductions
│   │   │   ├── base.py
│   │   │   └── concat.py
│   │   ├── output_modules
│   │   │   ├── base.py
│   │   │   ├── linear.py
│   │   │   └── none.py
│   │   ├── qkv_maps
│   │   │   ├── base.py
│   │   │   └── linear.py
│   │   └── taylor.py
│   ├── base.py
│   ├── classical.py
│   └── taylor.py
└── taylor.py

## Documentation
Visit [readthedocs](https://modular-transformer.readthedocs.io/) for documentation, API reference, and examples.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

## run example training
just train-example
```

You can run the ruff linter using `ruff --fix` and the ruff formatter using `ruff format`.
