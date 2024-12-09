# Plagiarism Checker

This repository contains a plagiarism checker that uses a fine-tuned BERT model to classify text as either human-written or AI-generated.

## Prerequisites

- Python 3.10 or higher
- `uv` (Universal Virtualenv) for managing virtual environments

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/debottamU31/plagiarism-checker-exercise.git
    cd plagiarism-checker-exercise
    ```

2. Follow the installation setup from [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. Run `uv sync` to create a virtual environment and install all the packages.
    - To install a new dependency and automatically add it to `pyproject.toml` use `uv add <package_name>`.

4. To activate the virtual environment run:
    ```bash
    source .venv/bin/activate
    ```

## Running the Code

1. **Train the model:**

    ```sh
    python plagiarism_checker.py
    ```

    This will train the BERT model using the dataset provided in [bigdataset.csv](http://_vscodecontentref_/1) and save the trained model as `my_model_v3_new.h5`.

2. **Classify text:**

    After training, you can classify text by running:

    ```sh
    python plagiarism_checker.py
    ```

    You will be prompted to enter the text you want to classify. The model will predict whether the text is human-written or AI-generated.

## Additional Information

- The dataset used for training should be in a CSV file named [bigdataset.csv](http://_vscodecontentref_/2) with columns `text` and `label`.
- The model and tokenizer are loaded from the `transformers` library, specifically using the `bert-base-uncased` model.

