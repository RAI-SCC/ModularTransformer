# Contributing

Welcome!
Currently contributions are only expected from members of the Junior Research Group for Robust and Efficient AI at the Scientific Computing Center Karlsruhe.
This contribution might be part of a master thesis or doctoral project.

## How to Contribute
...a new type of model or experiment.

0. **Join the github organization**

1. **Clone the repo**:
    ```bash
    git clone git@github.com:RAI-SCC/ModularTransformer.git
    ```

2. **Set up your enviroment**:
    ```bash
    python3 -m venv <path/to/your/venv>
    source <path/to/your/venv>/bin/activate
    pip install -U pip
    pip install -e ".[dev]"
    pre-commit install
    ```

3. **Create a branch** for your contribution:
    ```bash
    git checkout -b feature/<your-feature-name>
    ```

4. **Implement your contribution** taking into account the following guidelines:
    * All classes and functions have to be documented including all parameters and return types.
    * Please use American English for all comments and docstrings.
    * We use sphinx-autoapi to create an API reference. Please use the [NumPy Docstring Standard](https://numpydoc.readthedocs.io/en/latest/format.html).

5. **Use your contribution**:
    Add a demonstration of your contributions to the examples.

6. **Document your contribution**:
    Add a description of your contribution to the static example pages of the sphinx documentation.

7. **Add and commit your changes** with a clear and concise commit message.
    ```bash
    git commit -m "<your message>"

8. **Rebase to main**
    ```bash
    git checkout main
    git pull
    git checkout feature/<your-feature-name>
    git rebase main
    ```
    And resolve any conflicts.


9. **Push changes**
    ```bash
    git push
    ```

10. **Open a pull request**
    And wait for a review.

## Raise Issues

If there are questions or you encounter issues when using the Modular Transformer, please open an issue
