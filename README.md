# Location Mention Recognition (LMR) Fine-Tuning Script

## Overview

This repository contains a script designed for fine-tuning the GLiNER model to recognize and classify location mentions within text data. The script utilizes the Hugging Face `GLiNER` library and includes customizations for data processing, model training, and evaluation. It supports training with large, pre-trained GLiNER models, which have been specifically tailored for Named Entity Recognition (NER) tasks like location recognition.

## Requirements

- Python 3.8+
- CUDA-enabled GPU for model training
- Necessary Python packages (install via `requirements.txt`):
  - `transformers`
  - `torch`
  - `datasets`
  - `wandb`
  - `gliner`

## Environment Variables

This script requires several environment variables for managing resources and logging during model training:

- `TOKENIZERS_PARALLELISM`: Controls parallelism in tokenization to improve performance.
- `CUDA_DEVICE_ORDER`: Ensures that CUDA devices are initialized in a consistent order.
- `CUDA_VISIBLE_DEVICES`: Specifies which GPU to use for training.
- `WANDB_PROJECT`: Defines the project name for Weights and Biases logging.
- `WANDB_WATCH` & `WANDB_NOTEBOOK_NAME`: Used for logging with Weights and Biases.

## Dataset

The training and test datasets should be JSON files containing pre-processed location mention data. The script expects the following format:

```json
[
    {
        "tokens": [...],
        "ner": [[start_index, end_index, "LOC"]],
        "label": ["location"]
    }
]
```

### Dataset Paths:

- **Training Data**: `data/accepted_data/TrainCleaned.json`
- **Test Data**: `data/accepted_data/TestCleaned.json`

The script automatically updates the NER labels to "location".

Zindi dataset to train this model can be found [here](https://zindi.africa/competitions/microsoft-learn-location-mention-recognition-challenge/data)

## Usage

### Model Setup

The script can load several pre-trained GLiNER models for fine-tuning, including:

- `urchade/gliner_large-v2.1` (default)
- `urchade/gliner_small-v2.1`
- `urchade/gliner_medium-v2.1`

You can select a different model by changing the `GLiNER.from_pretrained()` method in the code.

### Fine-Tuning

The script fine-tunes the selected model on the provided dataset, using the `Trainer` class from the `GLiNER` library. Several training parameters, such as batch size, learning rate, and number of epochs, can be configured. Key parameters include:

- **Batch Size**: Set to 8 by default.
- **Learning Rate**: Default is `1e-6` with weight decay for regularization.
- **Epochs**: Number of epochs for training, set to 5.
- **Evaluation Strategy**: Evaluation is done periodically based on the number of steps.
- **Save Strategy**: Models are saved every `save_steps` interval, with the best model loaded at the end.

### Execution

To start training, run:

```bash
python <script_name>.py
```

### Logging

The script uses [Weights and Biases](https://wandb.ai/) for logging training progress, loss metrics, and saving checkpoints.

To use Weights and Biases, ensure you have set up a project by setting the environment variable `WANDB_PROJECT` and logging in via `wandb login`.

## Customization

- **Data Collator**: The data collator is configured to prepare labels for location mentions. You can modify this if you need to handle different types of entities.
- **Trainer Arguments**: You can fine-tune other hyperparameters in the `TrainingArguments` section such as gradient accumulation steps, learning rate scheduler, and evaluation strategy.
- **Model**: Various GLiNER models are available for specific use cases. To try a different pre-trained model, change the model's path in the `from_pretrained()` method.

## Example Output

The script will output:

1. Model summary.
2. Dataset size information.
3. Sample dataset entries.
4. Training logs with loss metrics.
5. Model evaluation results.

## Notes

- This script is optimized for large datasets and uses a constant learning rate for fine-tuning. Adjust the learning rate or use a different scheduler (e.g., cosine, linear) depending on your dataset and hardware.
- The script is configured to load the best model at the end of training based on evaluation loss.

---
