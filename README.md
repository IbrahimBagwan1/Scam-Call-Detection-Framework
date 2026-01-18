# Scam Call Detection

This project implements a scam call detection system using SMS transcripts. It leverages a pre-trained model from Hugging Face's Transformers library to classify messages as either scam or not.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries:
  - pandas
  - scikit-learn
  - transformers
  - datasets

### Usage

1. Place your SMS dataset in the same directory as `1_pretrain_sms.py`.
2. Run the script to train the model:
   ```bash
   python 1_pretrain_sms.py
   ```
3. The trained model will be saved in the `sms_pretrained_model` directory.

## Dataset

The dataset should be in CSV format with columns `transcript` and `labels`.

## License

This project is licensed under the MIT License.