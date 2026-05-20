# Breast Ultrasound CAD — CNN Classification with Hyperparameter Search & Interpretability

> Deep-learning pilot study comparing modern CNN architectures for benign-vs-malignant classification of breast ultrasound (BUS) images, with automated hyperparameter tuning and post-hoc model interpretation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-EE4C2C.svg)
![fastai](https://img.shields.io/badge/fastai-2.4-00A98F.svg)
![Optuna](https://img.shields.io/badge/Optuna-TPE-blueviolet.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-completed--2022-success.svg)

![image](https://user-images.githubusercontent.com/46795053/152703675-a13a24b9-3548-4356-9ffa-e0c0a9811605.png)

---

## Project Status

This is a **completed 2022 academic capstone project**. It is preserved here as a portfolio artifact and is not under active development. The training pipeline (`src/train.py`) is config-driven and can be reused or extended for other image-classification problems with minor adjustments.

## What This Project Demonstrates

- **Transfer learning** with multiple pre-trained CNN backbones (ResNet, DenseNet, etc.) via fastai
- **Domain-aware data augmentation** — geometric transforms constrained by ultrasound imaging conventions, plus MixUp for label-noise robustness
- **Hyperparameter optimization** with Optuna's Tree-structured Parzen Estimator (TPE) and median pruning
- **Modern training tricks**: discriminative learning rates, the 1-cycle scheduler (Smith, 2018), and progressive layer unfreezing
- **Experiment tracking** with Neptune (study metadata, metrics, and visualization artifacts logged per run)
- **Model interpretability** — both **LIME** (local surrogate explanations) and **GradCAM** (gradient-weighted class activation maps), so radiologists can audit *why* a prediction was made
- **End-to-end reproducibility** — a single JSON config drives the entire training + tuning loop

---

## Results

> Placeholder — fill in with best values from the Neptune study.

| Architecture | AUC | Accuracy | Precision | Recall | F1 |
| ------------ | --- | -------- | --------- | ------ | -- |
| ResNet18     | TBD | TBD      | TBD       | TBD    | TBD |
| ResNet50     | TBD | TBD      | TBD       | TBD    | TBD |
| DenseNet201  | TBD | TBD      | TBD       | TBD    | TBD |
| *Best model* | TBD | TBD      | TBD       | TBD    | TBD |

**Best configuration found by TPE search:**
- Learning rate: TBD
- Weight decay: TBD
- Batch size: TBD
- Freeze epochs: TBD

---

## Repository Structure

```
CapstoneBUS/
├── src/
│   └── train.py                    # Config-driven training + TPE hyperparameter search
├── configs/
│   └── example.json                # Example hyperparameter + tuning config
├── notebooks/
│   ├── example.ipynb               # End-to-end usage walkthrough
│   └── model_interpretation.ipynb  # LIME + GradCAM visualizations on a trained model
├── results/                        # (Output) plots, confusion matrices, top-loss images
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Install
```bash
git clone https://github.com/hall4jm/CapstoneBUS.git
cd CapstoneBUS
pip install -r requirements.txt
```

A CUDA-enabled GPU is strongly recommended. The project was developed against `fastai==2.4` and PyTorch 1.7–1.10; pinning matters because of fastai breaking changes after 2.4.

### 2. Prepare data
Organize images using fastai's `GrandparentSplitter` convention:
```
data_dir/
├── train/
│   ├── Benign/
│   └── Malignant/
└── val/
    ├── Benign/
    └── Malignant/
```

### 3. Configure
Edit [`configs/example.json`](configs/example.json) to point at your data and choose a backbone. The config has three sections:

- **`environ`** — paths, study name, number of trials
- **`hps`** — fixed hyperparameters (epochs, augmentation strength, image size, optimizer, backbone)
- **`tune`** — hyperparameters explored by Optuna (LR, weight decay, momentum, batch size, freeze epochs), each with an Optuna distribution and bounds

### 4. Add Neptune credentials
In `src/train.py`, replace `<PROJECT_NAME>` and `<API_TOKEN>` with your Neptune project / token (or remove the Neptune block if you don't want experiment tracking).

### 5. Run
```bash
python src/train.py -c configs/example.json
```

Outputs (per trial):
- `models/<backbone>_trial_<n>.pkl` — exported learner
- `img_dir/<backbone>_loss_trial_<n>.png` — training loss curve
- `img_dir/<backbone>_sched_trial_<n>.png` — 1-cycle LR schedule
- `img_dir/<backbone>_conf_matrix_trial_<n>.png` — confusion matrix
- `img_dir/<backbone>_top_losses_trial_<n>.png` — top-loss examples

The best trial's artifacts are uploaded to Neptune at the end of the study.

### 6. Interpret a trained model
Open [`notebooks/model_interpretation.ipynb`](notebooks/model_interpretation.ipynb), point it at an exported `.pkl`, and run the GradCAM / LIME cells to produce the saliency visualizations shown above.

---

## Background

According to the WHO, breast cancer is now the most common form of cancer globally (Breast cancer now most common form of cancer: WHO taking action, 2021). Over 280,000 new cases were estimated in the United States in 2021 alone, resulting in more than 43,000 deaths (U.S. Breast Cancer Statistics, 2021). Early detection is the single most important factor in reducing mortality (Sun et al., 2017): when potentially life-threatening lesions are caught early, patients have access to the widest set of effective treatment options.

To aid detection, doctors recommend regular mammograms and physical exams (Breast Cancer Early Detection and Diagnosis, 2021). Breast ultrasound (BUS) is a common alternative — non-radioactive, safe, non-invasive — and is often used to assess conspicuous masses without immediately resorting to biopsy.

### Why automate BUS interpretation?

Despite advances in imaging hardware, BUS interpretation still depends heavily on radiologist experience. Many BI-RADS features appear in both benign and malignant lesions, and the subjectivity of these characteristics leads to wildly varying assessments across radiologists. The positive-biopsy rate from BUS-driven referrals varies by as much as 51% between studies — meaning a large number of patients undergo unnecessary, invasive procedures.

Deep learning has shown strong results on many medical-imaging classification tasks, but most of these models fail to generalize to clinical settings. This project's aim is to evaluate whether a modern CNN-based CAD system, trained with appropriate augmentation and interpretability tooling, can produce a more standardized BUS assessment.

---

## Methodology

### Data Augmentation

CNNs are prone to memorizing small training sets rather than learning generalizable features. Two complementary augmentation strategies were used.

**1. Domain-aware geometric transforms.** Not every transformation makes sense for BUS imagery — these images are always captured with skin near the top and bone / deep tissue near the bottom, so a vertical flip would produce an image that violates the imaging protocol. Augmentations were restricted to:
- Rotations up to ±10°
- Zoom up to 1.5×
- Random crop to 224×224

**2. MixUp** (Zhang et al., 2017). A data-agnostic augmentation that linearly interpolates pairs of training images and their labels. MixUp is particularly powerful when labels are noisy — a real concern for BUS images, where the ground-truth labels were assigned by humans and are themselves subject to inter-rater variation.

### Models

Several pre-trained ImageNet backbones were evaluated:

![image](https://user-images.githubusercontent.com/46795053/152703177-3459d9ca-f3d8-46c5-8f63-7f9e197b95f9.png)

### Transfer Learning

Training from scratch is unstable on small medical-imaging datasets. Instead, each model was initialized with ImageNet-pretrained weights, the final classification layer was replaced with a fresh 2-class head, and training proceeded in two phases:

1. **Freeze + warm up the head.** Backbone weights are frozen; only the new classification head trains (for a number of epochs selected by Optuna).
2. **Unfreeze and fine-tune.** All layers train, but at different learning rates (see below).

### Discriminative Learning Rates

Early layers of a CNN learn generic features (edges, curves, corners) that transfer well from ImageNet, while deeper layers learn task-specific features that need more adjustment. fastai's discriminative learning rates assign progressively higher learning rates to deeper layers, so the backbone barely moves while the head and late layers adapt aggressively. This shortens training without sacrificing the value of the pretrained weights.

### Scheduler — 1-cycle policy

A fixed learning rate is rarely optimal across all of training. This project uses **cosine annealing under the 1-cycle policy** (Smith, 2018), which ramps the LR up from a starting value to a peak and then back down to zero across a single cycle. The 1-cycle policy is a strong general-purpose scheduler and substantially reduces training time relative to constant or step-decay schedules.

### Hyperparameter Search

The full hyperparameter space is explored by **Optuna's Tree-structured Parzen Estimator (TPE)** sampler with **median pruning** (5 startup trials, 15 minimum trials before pruning). Tuned hyperparameters:

- Learning rate (log-uniform, 1e-5 to 1e-1)
- Weight decay (log-uniform, 1e-5 to 1e-1)
- Momentum schedule (start / min / end momenta, ordered)
- Batch size (8 to 64, step 8)
- Freeze-phase epochs (0 to 25, step 5)

The objective is binary ROC AUC on the validation set.

---

## Model Interpretation

When a classification informs patient outcomes, radiologists need to trust *why* the model made a prediction. CNNs are notorious black boxes, so two interpretability methods were applied to the best-performing model:

**LIME** (Ribeiro et al., 2016) builds a local surrogate model by perturbing super-pixels in the input image and observing how the prediction probability changes. The result is a saliency mask over the regions that drove the prediction.

**GradCAM** uses the gradient of the predicted class with respect to the final convolutional feature maps, producing a class-discriminative heatmap that shows which spatial regions the network attended to.

Both are implemented in [`notebooks/model_interpretation.ipynb`](notebooks/model_interpretation.ipynb).

![image](https://user-images.githubusercontent.com/46795053/152703675-a13a24b9-3548-4356-9ffa-e0c0a9811605.png)

---

## Limitations & Future Work

- **Small dataset.** BUS images are scarce relative to natural-image datasets; the augmentation strategy and MixUp partially compensate but do not eliminate the issue. A larger multi-institution dataset would strengthen the results.
- **Binary classification only.** The model produces benign-vs-malignant predictions; clinical BI-RADS uses a 6-category scale. Extending to BI-RADS would require finer-grained labels.
- **Single-image input.** Radiologists use multiple views and patient context. A multi-view or multi-modal extension is a natural next step.
- **No external validation.** Performance is reported on a held-out validation split from the same dataset distribution. Generalization to other institutions' imaging hardware and protocols is not tested.
- **Interpretation quality.** LIME and GradCAM provide post-hoc rationalizations rather than guaranteed-faithful explanations. Inherently interpretable architectures would be a stronger guarantee.

---

## References

- Breast cancer now most common form of cancer: WHO taking action. (2021). World Health Organization.
- U.S. Breast Cancer Statistics. (2021). Breastcancer.org.
- Sun, Y. S., Zhao, Z., Yang, Z. N., Xu, F., Lu, H. J., Zhu, Z. Y., Shi, W., Jiang, J., Yao, P. P., & Zhu, H. P. (2017). Risk factors and preventions of breast cancer. *International Journal of Biological Sciences*, 13(11), 1387–1397.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD '16*.
- Smith, L. N. (2018). A disciplined approach to neural network hyper-parameters: Part 1 — learning rate, batch size, momentum, and weight decay. *arXiv:1803.09820*.
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond empirical risk minimization. *arXiv:1710.09412*.

---

## License

MIT — see [LICENSE](LICENSE).
