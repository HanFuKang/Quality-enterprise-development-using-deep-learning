# GNN-Enhanced Time Series Models for TFPLP Prediction

> Implementation code for:  
> **"A comprehensive analysis of digital inclusive finance's influence on high quality enterprise development through fixed effects and deep learning frameworks"**  
> *Scientific Reports, 2025* — DOI: [10.1038/s41598-025-14610-y](https://doi.org/10.1038/s41598-025-14610-y)

---

## Overview

This repository contains the implementation of GNN-enhanced time series models for predicting Total Factor Productivity (TFPLP) from Digital Inclusive Finance (DIF) panel data. Each folder corresponds to one hybrid model combining a Graph Convolutional Network (GCN) for cross-firm feature extraction with a classical time series backbone for sequential prediction.

---

## Repository Structure
```
.
├── GCN-LSTM/
├── GCN-BiLSTM/
├── GCN-GRU/
└── GCN-Transformer/
```

---

## Models

### GCN-LSTM
Combines a GCN feature extraction layer with a Long Short-Term Memory (LSTM) network. GCN aggregates neighbourhood information from the variable graph (nodes: DIF, 8 control variables, TFPLP; edges: statistical correlations and domain priors) and passes the extracted embeddings into the LSTM's input gate / forget gate / output gate architecture.

| RMSE | MAE | R² |
|------|-----|----|
| 0.2609 | 0.1565 | 0.8235 |

---

### GCN-BiLSTM
Extends the GCN-LSTM architecture with a bidirectional LSTM backbone, processing the time series in both forward and backward directions before concatenating hidden states. This captures both past and future temporal context in the economic panel data.

| RMSE | MAE | R² |
|------|-----|----|
| 0.2429 | 0.1519 | 0.8468 |

---

### GCN-GRU
Replaces the LSTM backbone with a Gated Recurrent Unit (GRU), which uses update and reset gates rather than the full LSTM cell. This offers reduced parameter count and faster training while retaining strong temporal modelling. GCN features are fed into the GRU input at each time step.

| RMSE | MAE | R² |
|------|-----|----|
| 0.2614 | 0.1544 | 0.8228 |

---

### GCN-Transformer
The highest-performing GCN variant. GCN-extracted graph embeddings are combined with the Transformer's self-attention and multi-head attention mechanism, enabling parallel modelling of complex temporal patterns. Represents a **77.89% improvement** over the baseline Transformer.

| RMSE | MAE | R² |
|------|-----|----|
| 0.0746 | 0.0412 | 0.9873 |

---

## Experimental Setup

| Setting | Detail |
|---------|--------|
| Dataset | 22,291 enterprise-year observations, 2011–2022 |
| Source | CSMAR database |
| Validation | 5-fold cross-validation, 80/20 train/test split |
| Graph construction | Pairwise variable correlations + domain knowledge |
| Evaluation metrics | R², RMSE, MSE, MAE |
| Statistical tests | Paired t-test, Cohen's d effect size |

---

## Citation
```bibtex
@article{wei2025digital,
  title     = {A comprehensive analysis of digital inclusive finance's influence 
               on high quality enterprise development through fixed effects 
               and deep learning frameworks},
  author    = {Wei, Dedai and Wang, Zimo and Kang, Hanfu and Sha, Xinye 
               and Xie, Yiran and Dai, Anqi and Ouyang, Kaichen},
  journal   = {Scientific Reports},
  volume    = {15},
  pages     = {30095},
  year      = {2025},
  doi       = {10.1038/s41598-025-14610-y}
}
```

---

## Contact

For questions regarding the implementation, please contact the corresponding authors:  
- Xinye Sha — xs2399@columbia.edu  
- Kaichen Ouyang — oykc@mail.ustc.edu.cn
