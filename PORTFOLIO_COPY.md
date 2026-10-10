# Portfolio / Resume Copy

## Recommended project title

**SpamShield — From-Scratch Email Spam Classifier**

## Tech stack

**Python, NumPy, NLP, Multinomial Naive Bayes, Bag-of-Words, Feature Engineering, Hyperparameter Tuning, Ablation Study, Matplotlib, Pytest, GitHub Actions**

## Resume version — English

- Built a **Multinomial Naive Bayes email spam classifier from scratch** in Python/NumPy, implementing tokenization, vocabulary construction, class priors, Laplace smoothing, log-likelihood inference, and evaluation without scikit-learn classifiers.
- Scaled the original experiment from **592 to 5,921 de-duplicated SpamAssassin emails** (**+5,329 records / +900.17% / ~10× data**) and built a deterministic stratified 60/20/20 train-validation-test pipeline.
- Evaluated **5 alpha settings, 5 vocabulary sizes, and 5 structural feature representations**; selected `alpha=2`, a **10,000-word vocabulary**, and URL-frequency features using validation F1.
- Achieved **97.97% held-out accuracy, 97.74% precision, 95.58% recall, and 96.65% F1** on 1,184 unseen test emails, with only **8 false positives and 16 false negatives**.
- Packaged the project as an installable `src/` repository with a cached experiment runner, saved model, CLI demo, automated tests, reproducible plots, and GitHub Actions CI.

## 简历版本 — 中文

- 使用 **Python/NumPy 从零实现 Multinomial Naive Bayes 垃圾邮件分类器**，独立完成文本分词、词表构建、类别先验、Laplace 平滑、对数似然推断及 Accuracy/Precision/Recall/F1 等指标计算，核心分类器不依赖 scikit-learn。
- 将原型数据由 **592 封扩展至 5,921 封去重 SpamAssassin 邮件**，新增 **5,329 封（+900.17%，约 10 倍数据规模）**，并构建可复现的 60/20/20 分层训练、验证与测试流程。
- 系统比较 **5 组 alpha、5 组词表规模及 5 类结构特征表示**，通过 validation F1 选出 `alpha=2`、**10,000 词词表**与 URL-frequency 特征。
- 在 **1,184 封完全独立测试邮件**上达到 **97.97% Accuracy、97.74% Precision、95.58% Recall、96.65% F1**，仅产生 **8 个 false positives 和 16 个 false negatives**。
- 完成工程化封装，包括 `src/` 模块化代码、缓存实验流程、模型保存、CLI 演示、单元测试、结果/图表自动生成及 GitHub Actions CI。

## GitHub Description

**From-scratch Multinomial Naive Bayes spam classifier on 5,921 de-duplicated SpamAssassin emails, with hyperparameter tuning, feature ablation, and 98.0% held-out accuracy.**

## Suggested repository name

`spamshield-naive-bayes`

## Suggested GitHub topics

`python`, `numpy`, `nlp`, `naive-bayes`, `spam-classification`, `email-classification`, `machine-learning`, `feature-engineering`, `ablation-study`, `from-scratch`, `portfolio-project`

## Interview-ready numbers

- Full dataset: **5,921 emails** = 4,109 ham + 1,812 spam.
- Prototype → full dataset: **592 → 5,921**, +5,329 records, **+900.17% / ~10×**.
- Split: **3,553 train / 1,184 validation / 1,184 test**.
- Selected configuration: **alpha = 2**, **vocabulary = 10,000**, **words + URL-frequency feature**.
- Validation F1: words only **95.30%** → words + URL **95.84%** (**+0.54 pp**).
- Held-out test: **97.97% accuracy / 97.74% precision / 95.58% recall / 96.65% F1**.
- Prototype → full-data held-out performance: Accuracy **+2.21 pp**; F1 **+3.69 pp**.

## Safe wording for interviews

Say that the project **scaled the dataset by about 10×, expanded the model-selection search for the larger corpus, and improved held-out performance**. Do not attribute the entire test improvement to URL features; their measured validation contribution is +0.54 percentage points.
