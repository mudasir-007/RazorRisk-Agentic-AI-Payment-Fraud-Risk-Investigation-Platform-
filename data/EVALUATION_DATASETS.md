# Evaluation Dataset

RazorRisk uses **one synthetic dataset as the model source of truth**. Both ML branches are trained from the same generated transaction population and are evaluated on the same held-out users/transactions.

## Synthetic RazorRisk dataset

The generator creates:

- transaction-level behavioral features for XGBoost;
- User/Device/IP/Merchant relationships for GraphSAGE;
- named fraud rings;
- benign shared-infrastructure look-alikes;
- adversarial cases such as structuring, fan-out laundering, no-shared-infrastructure fraud, low-and-slow fraud, cold-start fraud, and account takeover.

Latest reproducible run (`seed=42`, 3,000 requested normal users, 30,000 requested baseline transactions): **31,048 transactions / 3,450 users** after scenario injection, including **293 fraud-labelled transactions**. The user-level split used for the current evaluation contains **21,830 training transactions / 215 fraud** and **9,218 test transactions / 78 fraud**.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Tabular / XGBoost | **0.997750** | **0.951347** | 0.755102 | **0.948718** | 0.840909 |
| GraphSAGE / GNN | **0.997178** | **0.893752** | 0.411765 | 0.897436 | 0.564516 |
| Learned stacker | **0.998784** | **0.953931** | **0.804348** | **0.948718** | **0.870588** |

The stacker is trained from paired XGBoost/GNN predictions for the same synthetic transactions, plus normalized shared-device/shared-IP evidence. It uses `class_weight="balanced"` so the rare fraud class is not overwhelmed by the majority class.

Run:

```bash
python tests/evaluate_models.py --dataset synthetic
python tests/evaluate_models.py --dataset synthetic --retrain
```

## External datasets

The ULB/Kaggle `creditcard.csv` dataset is **not part of the RazorRisk model pipeline or evaluation contract**. Its anonymized PCA features and lack of User/Device/IP/Merchant identities do not match the project's feature and graph domain. It may be retained as a research artifact, but its metrics must not be presented as RazorRisk system metrics.
