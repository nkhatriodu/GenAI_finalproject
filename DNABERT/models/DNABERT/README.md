# Evaluation Summary – Fine-Tuned DNABERT (Epoch 4)
## To Do
- Upload to Hugging Face
- Update instructions for inference
- Include sample dataset for inference

## Model Checkpoint
- Reloaded from: `./runs/dnabert_fullft_es/best_model` (Epoch 4)
- Tokenizer and model were reloaded from the checkpoint and confirmed functional.

---

## Validation Metrics (Default Threshold = 0.5)

| Metric     | Value     |
|------------|-----------|
| AUC        | 0.9024    |
| Accuracy   | 0.8476    |
| Precision  | 0.8430    |
| Recall     | 0.9929    |
| F1 Score   | 0.9125    |

---

## Threshold Optimization

- Thresholds tested: 0.10 to 0.90 (step=0.05)
- Optimal threshold selected by max F1 (with tie-breaker on recall → precision)

### Best Threshold: **0.50**
| Metric     | Value     |
|------------|-----------|
| Accuracy   | 0.8476    |
| Precision  | 0.8430    |
| Recall     | 0.9929    |
| F1 Score   | 0.9125    |
| TP / FP / TN / FN | 2784 / 518 / 1421 / 20 |

---

## ROC Curve

- AUC of 0.9024 confirms strong separability between benign/pathogenic classes.
- Curve plotted from predictions on the held-out validation set.

---

## Precision-Recall Curve

- AUPRC also confirms strong precision across a wide range of recall values.
- Useful due to class imbalance (pathogenic is minority class).

---

## Files & Usage

- This model is now ready for downstream deployment or use in inference.
- Example usage to reload:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./runs/dnabert_fullft_es/best_model")
tokenizer = AutoTokenizer.from_pretrained("./runs/dnabert_fullft_es/best_model", use_fast=False)
```

---

## Notes

- This summary corresponds to full fine-tuning (all layers) of `zhihan1996/DNA_bert_6`.
- Labels: 0 = Benign, 1 = Pathogenic
- Training performed using 6-mers from variant alt sequence (`text_alt_k6`)

---

**Author:** Terry Stilwell  
**Course:** CS895 – Generative AI for Scientific Discovery  
**Project:** Fine-Tuning Genomic Foundation Models for Variant Effect Prediction
