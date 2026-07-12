# A machine learning framework for predicting and modulating condition-dependent protein phase separation

This is the official implementation of the paper **"A machine learning framework for predicting and modulating condition-dependent protein phase separation"**.
<br>A preprint version of our paper is available on bioRxiv.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.12.28.696755-b31b1b.svg)](https://doi.org/10.64898/2025.12.28.696755) [![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-LLPSense-yellow.svg)](https://huggingface.co/spaces/HugMinjun/LLPSense)

**Authors:**
Jangwon Bae<sup>†\*</sup>, Minjun Kang<sup>†</sup>, Donghyuk Lee, Kuk-Jin Yoon<sup>\*</sup>, and Yongwon Jung<sup>\*</sup>

<sup>†</sup> These authors contributed equally to this work.  
<sup>\*</sup> Corresponding authors.

---

## 📖 Abstract
Protein phase separation is a fundamental process in organizing membraneless organelles and is implicated in a wide range of pathological conditions. Crucially, rather than being an intrinsic feature of specific proteins, **phase separation is a condition-dependent phenomenon** governed by environmental parameters, including protein concentration, temperature, and solvent composition. However, most existing machine-learning models infer phase separation propensity from amino-acid sequences alone, failing to capture these context-dependent behaviors.

Here, we present **LLPSense**, a machine-learning framework that integrates pre-trained protein language model embeddings with environmental parameters to achieve condition-integrated prediction with substantially improved accuracy. We demonstrate LLPSense's predictive power and applications through multiple experimental validations:
1.  **Identification of unrecognized phase-separating protein and its complex behavior:** LLPSense newly identified SGTA as a phase-separating protein within datasets formerly classified as negative. It accurately predicted the protein’s complex, temperature-dependent reentrant phase behavior, which we subsequently confirmed experimentally.
2.  **Single-residue resolution mapping of mutational landscapes:** We performed in silico mutagenesis of alpha-synuclein, a protein associated with Parkinson's disease. Experimental validation confirmed the model's predictive accuracy in identifying high-impact variants, representing a significant advancement in bidirectional mutational analysis that encompasses both the enhancement and suppression of phase separation at single-residue resolution.
3.  **Reprogramming of environment-responsive phase profiles:** Through model-guided mutagenesis, we successfully inverted the phase behavior of UBQLN4 from a lower critical solution temperature (LCST) to an upper critical solution temperature (UCST) regime. This highlights LLPSense as a powerful tool for the rational design of sequence modifications to precisely modulate environment-responsive properties.

Collectively, LLPSense provides a robust computational tool for interrogating the condition-dependent landscape of protein phase separation, enabling mechanistic studies of disease-associated phase separation as well as the rational design of programmable condensates.

> 🤗 <b>Try our interactive demo — no setup required:</b> <a href="https://huggingface.co/spaces/HugMinjun/LLPSense"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-LLPSense-yellow.svg" valign="middle"></a>

---

## ⚙️ Installation

We highly recommend using Docker to set up the environment. For Windows users, WSL2 (Windows Subsystem for Linux) is supported and recommended for running Docker. Please ensure that Docker (and WSL2 for Windows users) is installed and configured before proceeding with the steps below. Our code has been tested on Ubuntu 22.04.3 LTS (Python 3.10.14, CUDA 11.8). No separate installation is required for LLPSense itself; the setup involves preparing the environment and dependencies, which typically takes under an hour.

### System Requirements and Runtime
The computational requirements for LLPSense depend on whether you are extracting features or running inference:

* Feature Extraction (ProtT5): To process protein sequences up to 3,000 AA in length, an NVIDIA GPU with at least 16GB VRAM is recommended. (approx. 100 ms/protein on RTX A6000, varies depending on the protein length)

* Inference: Once features are extracted, the LLPSense model is highly efficient and can run on consumer-grade GPUs (< 8GB VRAM) or even on a CPU. (34 ms/condition on RTX A6000, benchmarked on the human proteome)
<details>
<summary><b>Click to expand — Docker setup & install commands</b></summary>

<br>

```bash
# 1. Pull the base image
docker pull pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# 2. Create and start the container (Choose one of the options below)
# Option 2-A. GPU - Recommended
docker run -it -d --gpus all --name llpsense --shm-size=256g --privileged -v "${PWD}:/workspace" pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
docker attach llpsense
# Option 2-B. CPU - Inference only
docker run -it -d --name llpsense --shm-size=16g --privileged -v "${PWD}:/workspace" pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
docker attach llpsense

# 3. Install system dependencies
apt update && apt-get install -y software-properties-common
apt-get install -y git nano curl wget unzip cmake tmux gparted htop net-tools aria2 rclone ffmpeg libglib2.0-0 libgl1-mesa-glx libx11-dev xorg-dev libglu1-mesa-dev libglew-dev libglm-dev libboost-all-dev pybind11-dev

# 4. Clone the repository
cd /workspace
git clone https://github.com/NearNiah/LLPSense
cd LLPSense

# 5. Install Python requirements
pip install -r requirements.txt
```

</details>

---

## ▶️ Quick Start
We provide preprocessed features for **α-synuclein**, including our LLPS-modulated variants and pathogenic variants with LLPS propensity confirmed in the literature.

### Instruction
Running the command below will generate a prediction spreadsheet.
```bash
python LLPSense_standalone.py standalone.exp_task=predict data.test_file=data/processed/alphasyn.npz expname=alphasyn
```
The result will be saved to `outputs/results/LLPSense_alphasyn/predict.xlsx`.

### Expected Output
The output file (`predict.xlsx`) should contain:

* LLPS probabilities for **our LLPS-modulated variants** under standard conditions (**Supplementary Table 1**).

* LLPS probabilities for **pathogenic variants** under standard conditions (**Supplementary Figure 7**).

### Expected Runtime
**Typical runtime: ~5 seconds or less** on a standard desktop computer. 
Runtime primarily depends on CPU/GPU speed, disk I/O, and whether additional models or assets are loaded during inference.

---


## 💽 Data Preparation (Your own protein)

To apply the model to your own proteins, you need to format the sequences and experimental conditions, followed by feature extraction.

<details>
<summary><b>Click to expand — file formatting & preprocessing</b></summary>

<br>

### 1. File Formatting
Prepare an Excel file (`.xlsx`) containing the fields below and place it in `data/raw`.
An example template is provided: [Data_template.xlsx](data/raw/Data_template.xlsx).

| Column | Description | Range / Condition |
| :--- | :--- | :---: |
| **Protein_name** | Name or ID of the protein (Optional) | - |
| **Sequence** | Amino acid sequence (1-letter code) | - |
| **Conc** | Protein concentration [µM] | 0 - 1000 |
| **pH** | pH level | 0 - 14 |
| **temp** | Temperature [°C] | 0 - 60 |
| **MgCl2** | Concentration of MgCl2 [mM] | 0 - 50 |
| **NaCl** | Concentration of NaCl [mM] | 0 - 2000 |
| **KCl** | Concentration of KCl [mM] | 0 - 1000 |
| **PEG300-1k** | Percentage of PEG 300-1k [%] | 0 - 50 |
| **PEG3k-6k** | Percentage of PEG 3k-6k [%] | 0 - 50 |
| **PEG8k-20k** | Percentage of PEG 8k-20k [%] | 0 - 50 |
| **Ficoll** | Percentage of Ficoll [%] | 0 - 50 |
| **Dextran -40** | Dextran with MW ≤ 40 kDa [%] | 0 - 50 |
| **Dextran 70-** | Dextran with MW ≥ 70 kDa [%] | 0 - 50 |
| **Glycerol** | Percentage of Glycerol [%] | 0 - 10 |
| **label** | LLPS propensity score (Target) | 0 or 1 |
| **cluster/group** | Cluster ID for data splitting/evaluation | - |

> **Note:** `label` and `cluster/group` are only required for training or evaluation tasks. For prediction on new proteins, these columns can be left empty.

### 2. Preprocessing
Run the following scripts to process the raw file and extract features.

#### Step 1: Process Raw Data
Choose one of the following options based on your task.

**Option A: Standard Processing (For Prediction/Training)**
Converts raw `.xlsx` data to `.npz` format for general tasks. The `.xlsx` file should be located in `data/raw`. We use this preprocessing pipeline for [A. Evaluation](#a-evaluate-model-performance), [B. Prediction of LLPS](#b-predict-llps-probability), and [C. Condition Screening](#c-condition-screening).

```bash
python -m preprocess.process_db --srcfilename=Data_template.xlsx
```

**Option B: Mutation Processing**
Generates a `.npz` file containing **all possible single-point mutations** for the input `.xlsx` proteins. We use this preprocessing pipeline for [D. Mutation Mapping](#d-mutation-mapping) and [E. Mutation Screening](#e-identification-of-mutations-modulating-condition-dependency).
```bash
python -m preprocess.process_mut --srcfilename=Data_template.xlsx
```

#### Step 2: Extract Features
Extract [ProtTrans (T5)](https://github.com/agemagician/ProtTrans) embeddings or engineered features.

For Language Model features (T5):
```bash
python -m preprocess.extract_feat --feature_type=t5 --srcfilename=Data_template.npz
```
For Engineered features:
```bash
python -m preprocess.extract_feat --feature_type=eng --srcfilename=Data_template.npz 
```
*See [preprocess/extract_feat.py](preprocess/extract_feat.py) for implementation details.*

</details>

---

## 🚀 Usage

<details>
<summary><b>Click to expand — full usage guide (Prediction, Screening, Mutation, SHAP)</b></summary>

<br>

### 1. LLPSense (Sequence + Condition)
**LLPSense** predicts LLPS propensity given a **protein sequence** under **specific environmental conditions**.

**Key Arguments:**
* `standalone.exp_task`: Task mode (default: `predict`; options: `test`, `screen`, `mut`, `mut_screen`).
* `data.test_file` / `data.screen_file` / `data.mut_file`: Path to the `.npz` file generated in the Data Preparation step (usually in `data/processed`).
* `expname`: Name of the experiment (defines the output directory; default: `default`).

#### A. Evaluate Model Performance
Predicts scores for labeled data and calculates performance metrics. We used this function to evaluate our model's performance.
```bash
python LLPSense_standalone.py standalone.exp_task=test data.test_file=data/processed/Dataset_test.npz expname=test
```

#### B. Predict LLPS Probability
Predicts scores for new proteins/conditions (no labels required). We used this function to predict LLPS probability given proteins under specified condition.
* Specify `data.test_file`.
```bash
python LLPSense_standalone.py standalone.exp_task=predict data.test_file=data/processed/Data_template.npz expname=predict_template
```

#### C. Condition Screening
Screens LLPS behavior across a range of conditions (e.g., Temp, pH, Salt). We used this function for screening phase separating candidates and condition-dependency profiling.
* Change `standalone.screen` to `temp`, `conc`, `pH`, or `nacl`.
* Specify `data.screen_file`.
* (Optional) Modify base conditions in `preprocess/misc.py` (`base_cond_screen`).

Example: Screening temperature dependence
```bash
python LLPSense_standalone.py standalone.exp_task=screen standalone.screen=temp data.screen_file=data/processed/Data_template.npz expname=screen_temp_template
```

#### D. Mutation Mapping
Maps the predicted LLPS probabilities of single-point mutations. We used this function for predicting LLPS probability of α-synuclein variants.
* Specify `data.mut_file`.
```bash
python LLPSense_standalone.py standalone.exp_task=mut data.mut_file=data/processed/Data_template.npz expname=mut_template
```

#### E. Identification of Mutations Modulating Condition-Dependency
Identifies mutations that result in the **most significant negative/positive shift in the gradient** of LLPS probability with respect to environmental conditions. We used this function to modulate condition-dependent phase behavior of UBQLN4.
* `standalone.num_mut`: Number of top-ranked single mutations to recommend, selected based on the magnitude of slope change (default: 1).
* Change `standalone.screen` to `temp`, `conc`, `pH`, or `nacl`.
* Specify `data.mut_file`.
* Set `standalone.mut_direction` to `negative` or `positive`.
* (Optional) Modify base conditions in `preprocess/misc.py` (`base_cond_mutation`).

Example: Mutation screening under temperature gradient (LCST to UCST)
```bash
python LLPSense_standalone.py standalone.exp_task=mut_screen standalone.screen=temp data.mut_file=data/processed/Data_template.npz expname=mut_screen_temp_template
```
If you want to reproduce our UBQLN4 result, you can use our custom script that iteratively mutates protein and modulate condition-dependent phase behavior.
> **⚠️ Pre-requisite:** You need to first preprocess `UBQLN4.xlsx` in `data/raw` following the [instruction](#-data-preparation-your-own-protein).
```bash
sh ./scripts/mut_screen_neg.sh UBQLN4 10 3  # mutate 3 times and find top-10 single mutations at each time.
```

#### F. SHAP Analysis and Channel Reduction
SHAP analysis helps interpret which T5 feature channels or conditions contribute most to LLPS predictions. In this workflow, you first generate prediction outputs, then convert the exported SHAP scores into an index map, and finally retrain or reuse a compact channel-reduced model.

* Step 1: prepare the T5-based processed datasets for the human proteome and LLPSense benchmark.
* Step 2: run prediction to export SHAP results into the corresponding output directory.
* Step 3: move `predict_shap_data.xlsx` into `shap_analysis/` and build the feature-importance index map.
* Step 4: optionally search for the best channel-reduction setting with Optuna.
* Step 5: run inference with the trained or provided squeezed model.

```bash
# Step 1. follow preprocessing step to get T5 feature of human_proteome / LLPSense Dataset

# Step 2. run prediction and get shap analysis file under outputs/results/LLPSense_human_shap or outputs/results/LLPSense_llpsdb_shap
python LLPSense_standalone.py standalone.exp_task=predict data.test_file=data/processed/human_proteome.npz expname=human_shap
python LLPSense_standalone.py standalone.exp_task=predict data.test_file=data/processed/Dataset_LLPSense.npz expname=llpsdb_shap

# Step 3. move and rename predict_shap_data.xlsx to shap_analysis directory.
# We already provide human_shap.xlsx and llpsdb_shap.xlsx from step 2.
# Run the following command to make the feature-importance index map.
python -m shap_analysis.shap_sampling

# Step 4. (optional) get the optimal parameter of LLPSense with channel reduction using optuna.
python train_LLPSense.py --config-name=LLPSense_human_topk method.topk=512 method.mode=optuna

# Step 5. run inference using the trained model; a 1024->64 channel-squeezed model is provided at models/LLPSense_llpsdb_top64_shap.pkl
python LLPSense_standalone.py --config-name=LLPSense_llpsdb_topk standalone.exp_task=predict data.test_file=data/processed/alphasyn.npz expname=alphasyn_top64_shap
```

### 2. LLPSeq (Sequence Only)
**LLPSeq** predicts LLPS propensity based **solely on protein sequence**. We benchmark against the [PSPire](https://www.nature.com/articles/s41467-024-46445-y) dataset.

**Key Arguments:**
* `data.test_file1`: Path to the processed **ID-PSPs** test set from [PSPire [1]](https://www.nature.com/articles/s41467-024-46445-y).
* `data.test_file2`: Path to the processed **noID-PSPs** test set from [PSPire [1]](https://www.nature.com/articles/s41467-024-46445-y).
* `expname`: Name of the experiment (defines the output directory; default: `default`).

> **⚠️ Pre-requisite:** You need to first preprocess `PSPire_test_*.xlsx` in `data/raw` following the [instruction](#-data-preparation-your-own-protein).

#### A. Evaluate Model Performance
```bash
# Evaluate LLPSeq (Engineered Features)
python train_LLPSeq.py --config-name LLPSeq_eng method.mode=train

# Evaluate LLPSeq (pLM-based Features)
python train_LLPSeq.py --config-name LLPSeq method.mode=train
```

#### B. Predict LLPS Probability
```bash
# Predict with LLPSeq (pLM-based Features, trained with training dataset)
python LLPSeq_standalone.py --config-name LLPSeq data.test_file1=data/processed/Data_template.npz

# Predict with LLPSeq_full (pLM-based Features, trained with full dataset)
python LLPSeq_standalone.py --config-name LLPSeq_full data.test_file1=data/processed/Data_template.npz
```

</details>

---

## ⚡ Training
We use [Optuna](https://optuna.org/) for hyperparameter optimization and training.

<details>
<summary><b>Click to expand — LLPSense & LLPSeq training</b></summary>

<br>

### 1. LLPSense Model
**Key Arguments:**
* `data.train_file`: Path to the processed training dataset. For reproducibility, we provide `Dataset_LLPSense.npz` in `data/processed`.
* `expname`: Name of the experiment (defines the output directory; default: `default`).

> **⚠️ Pre-requisite:** You need to preprocess `Dataset_LLPSense.npz` following the [instruction](#-data-preparation-your-own-protein).

```bash
python train_LLPSense.py method.mode=optuna
```

### 2. LLPSeq Model
**Key Arguments:**
* `data.train_file`: Path to the processed **PSPire** training set.
* `expname`: Name of the experiment (defines the output directory; default: `default`).

> **⚠️ Pre-requisite:** You need to first preprocess `PSPire_train.xlsx` in `data/raw` following the [instruction](#-data-preparation-your-own-protein).

```bash
# Train LLPSeq (Engineered Features)
python train_LLPSeq.py --config-name LLPSeq_eng method.mode=optuna check.save_model=True

# Train LLPSeq (pLM-based Features)
python train_LLPSeq.py --config-name LLPSeq method.mode=optuna check.save_model=True
```

</details>

---

## 📊 Benchmarks

**Sequence and Condition-based Prediction:**

| Model | Accuracy | AUROC | AUPRC |
| :--- | :---: | :---: | :---: |
| [Droppler [2]](https://academic.oup.com/bioinformatics/article/37/20/3473/6275261) (Bioinformatics '21) | 62 | 67 (64*) | 59 |
| **LLPSense (Ours)** | **68** | **77** | **74** |

<sup>*</sup> *Official AUROC reported in the Droppler paper. Values in the main column are obtained by training on our novel expanded dataset from scratch. Note that training Droppler on our dataset leads to improved results.*

---

## ⭐ Citation
If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{bae2025machine,
  title={A machine learning framework for predicting and modulating condition-dependent protein phase separation},
  author={Bae, Jangwon and Kang, Minjun and Lee, Donghyuk and Yoon, Kuk-Jin and Jung, Yongwon},
  journal={bioRxiv},
  pages={2025--12},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

---

## 🙏 Acknowledgements
Our project was deeply inspired by and built upon previous studies. We explicitly thank the authors and contributors of the following works:
* [ProtTrans](https://ieeexplore.ieee.org/document/9477085) (IEEE TPAMI '21)
* [DeePhase](https://www.pnas.org/doi/10.1073/pnas.2019053118) (PNAS '21)
* [PSPire](https://www.nature.com/articles/s41467-024-46445-y) (Nat. Commun. '24)
* [PSPHunter](https://www.nature.com/articles/s41467-024-46901-9) (Nat. Commun '24)
* [Droppler](https://academic.oup.com/bioinformatics/article/37/20/3473/6275261) (Bioinformatics '21)

---

## 📧 Contact
We warmly welcome the application of LLPSense to a wide range of research fields. Our goal is to make this tool as accessible and impactful as possible for the scientific community.

If you encounter any challenges during implementation, have questions regarding the project, or are interested in potential collaborations, please feel free to reach out. You can get in touch by opening an issue in this repository or by contacting the authors directly:
* **Jangwon Bae**: [baejang1@kaist.ac.kr](mailto:baejang1@kaist.ac.kr)
* **Minjun Kang**: [kmmj2005@kaist.ac.kr](mailto:kmmj2005@kaist.ac.kr)
