# Attention-Based-Drug-QA-Model
# Drug Attention QA Model

A biomedical question-answering system focused on drug information, built using BioBERT and PyTorch.

## Overview

This project implements a specialized question-answering model for drug-related queries. The model uses BioBERT (Bidirectional Encoder Representations from Transformers pre-trained on biomedical text) to understand and answer questions about medications, their side effects, dosages, and uses.

## Features

- Fine-tuned BioBERT model for biomedical question answering
- Support for multiple dataset sources (BioASQ, DrugEHRQA, synthetic data)
- Interactive Q&A interface for drug information queries
- Evaluation metrics including Exact Match and F1 Score
- Built-in context about common medications
- Confidence scoring for answers

## Requirements

- Python 3.6+
- PyTorch
- TensorFlow
- Transformers
- Pandas
- NumPy
- scikit-learn
- NLTK
- tqdm

Install dependencies with:
bash
pip install torch tensorflow transformers pandas numpy scikit-learn nltk tqdm matplotlib


## Dataset

The model can work with several biomedical QA datasets:

1. *BioASQ*: A biomedical semantic indexing and question answering challenge
2. *DrugEHRQA*: A dataset focusing on drug-related questions in electronic health records
3. *Synthetic Dataset*: A fallback option that generates realistic drug-related QA pairs

The implementation attempts to download these datasets in order of preference and falls back to synthetic data generation if necessary.

## Model Architecture

The system uses the BioBERT v1.1 model, which is a version of BERT pre-trained on biomedical text. Key components include:

- BioBERT base model with question-answering head
- Custom dataset processing for handling answer spans
- AdamW optimizer with linear learning rate scheduler
- Context-based answer extraction

## Usage

### Training the Model

python
# Cells 1-5 handle model setup and training
# The model will be saved to ./biobert_drug_qa


### Evaluating the Model

python
# Cell 6 evaluates the model on the test set
# Metrics include Exact Match and F1 Score


### Sample Questions

python
# Cell 7 demonstrates the model with predefined examples


### Interactive Mode

python
# Cell 8 provides an interactive interface
interactive_qa()  # Run this function to interact with the model


## Example Questions

The model can answer questions like:
- "What are the side effects of Aspirin?"
- "How should I take Metformin?"
- "Can antibiotics treat viral infections?"
- "What is Lisinopril used for?"

## Future Improvements

- Expand the default context with information about more medications
- Implement negation handling for more accurate answers
- Add support for multi-document question answering
- Enhance the model with entity linking to medical knowledge bases

## License
