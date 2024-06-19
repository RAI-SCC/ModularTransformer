alias t := train-example

train-example:
    #!/usr/bin/env bash
    source venv/bin/activate
    set -euxo pipefail
    python examples/taylor/train.py

mypy:
    #!/usr/bin/env bash
    source venv/bin/activate
    set -euxo pipefail
    mypy modular_transformer
