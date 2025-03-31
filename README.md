# unlp_shared_task
The Fourth Ukrainian NLP Workshop (UNLP 2025) organizes a Shared Task on Detecting Social Media Manipulation.
In this shared task, the objective is to perform multilabel classification on Ukrainian and Russian social media posts to detect various manipulation techniques (e.g., "straw_man", "loaded_language", "euphoria").

Initial Approach: Ensemble Model

Data Preprocessing:
The pipeline cleans the raw text by lowercasing, removing punctuation/digits, normalizing whitespace, and optionally removing stopwords (combining Russian and a custom Ukrainian list). Labels are then one-hot encoded. Additionally, contextual augmentation is applied using ContextualWordEmbsAug (with xlm-roberta-base) to generate extra training samples.

Model Training & Ensemble:
An ensemble of three transformer models is trained: XLM-RoBERTa-large (with increased dropout for regularization), bert-base-multilingual-cased, and DeBERTa-v3-base. Predictions from these models are combined using a weighted average, and optimal thresholding is determined using BayesSearchCV to maximize macro F1.

Outcome:
Although the ensemble approach was explored, the macro F1 score was unsatisfactory.

Refined Approach: XLM-RoBERTa-base Only
Due to poor ensemble performance, the training strategy was shifted to use only the XLM-RoBERTa-base model. The same data preprocessing and augmentation steps are applied, but now only XLM-RoBERTa-base is fine-tuned using the Hugging Face Trainer.

Test Set Inference:
After training, the model is used to predict on the test set. The raw logits are converted to probabilities using a sigmoid, and then binarized (with an optimized threshold, if needed) to generate the final predictions for submission.

This streamlined pipeline leverages XLM-RoBERTa-base for improved robustness in a multilingual setting, addressing the shortcomings observed with the ensemble approach.
