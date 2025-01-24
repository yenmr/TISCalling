# TISCalling - TIS Prediction Using Machine Learning

TISCalling is an open-source project dedicated to predicting non-canonical Translation Initiation Sites (TIS) in transcriptomic data across various species. The project uses advanced machine learning techniques to develop a robust classifier that accurately identifies non-canonical TIS locations based on sequence data and other relevant features.

## Table of Contents

1. [Introduction](#introduction)
2. [Features and Dataset](#features-and-dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [License](#license)

## Introduction
Translation Initiation Sites (TIS) are critical for protein synthesis, and accurately identifying non-canonical sites can advance our understanding of gene expression regulation across different species. This project leverages machine learning to predict non-canonical TIS, emphasizing scalability and accuracy.

## Features and Dataset

### Features
- **Kozake:** Features based on the Kozak sequence context for translation initiation.
- **Noderer:** Features derived from Noderer score metrics.
- **PWM:** Position Weight Matrix features based on TIS and their neighboring sequences.
- **TIS codon usage:** Codon usage statistics specific to TIS.
- **CDS:** Coding sequence-specific features.
- **RNA:** RNA secondary structure and accessibility features.
- **Kmer:** k-mer frequency features.

### Dataset

This project uses structured datasets to train and evaluate the TIS prediction models. The dataset consists of:

- **Inputs for Model Training:**
  - Labeled examples of TIS (both positive and negative) in TSV format.
  - Transcript sequences in TSV format, providing the context for TIS identification.

- **Inputs for TIS Prediction:**
  - DNA sequences in FASTA format for identifying TIS in new data.

## Example Files

The following example files are automatically generated in the `input_data/` folder after installation:
*
  - `Tomato_TP_and_TN_data.txt`: Contains true positive and true negative TIS examples.
  - `Tomato_seq.txt`: Includes transcript sequences used for feature extraction.
  - `predict_input.fa`: FASTA file used as input for TIS prediction.

These files serve as a starting point for processing and feature generation during model training and prediction.  - `Tomato_TP_and_TN_data.txt`: True positive and true negative TIS sites.
  - `Tomato_seq.txt`: Transcript sequences.
  - `predict_input.fa`: Input data for prediction in FASTA format.

## Installation

```bash
# Clone the repository
git clone https://github.com/yenmr/TIScalling.git
cd TIScalling

# Create Conda environment
conda create -n TIScalling python=3.10
conda activate TIScalling

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Preprocess Data

#### Prepare PWM and Codon Usage Table for generating features

```bash
# Prepare PWM 
python script/PWM_generator.py -seq input_data/Tomato_seq.txt -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -filename PWM_Sl_5UTRATG.txt -TP_type 2
python script/PWM_generator.py -seq input_data/Tomato_seq.txt -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -filename PWM_Sl_CDSATG.txt -TP_type 3
python script/PWM_generator.py -seq input_data/Tomato_seq.txt -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -filename PWM_Sl_5UTRnonATG.txt -TP_type 4
python script/PWM_generator.py -seq input_data/Tomato_seq.txt -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -filename PWM_Sl_CDSnonATG.txt -TP_type 5

# -TP_type specifies the TIS category:
# 2: 5UTR_ATG, 3: CDS_ATG, 4: 5UTR_nonATG, 5: CDS_nonATG

# Output Directory: The generated PWM files will be stored in the `PWM_library` folder.

# Prepare Codon Usage Table
python script/coden_usage_generator.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TP_type 2 -filename TIS_codon_Sl_5UTRATG.txt -sp Sl
python script/coden_usage_generator.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TP_type 3 -filename TIS_codon_Sl_CDSATG.txt -sp Sl
python script/coden_usage_generator.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TP_type 4 -filename TIS_codon_Sl_5UTRnonATG.txt -sp Sl
python script/coden_usage_generator.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TP_type 5 -filename TIS_codon_Sl_CDSnonATG.txt -sp Sl

# -TP_type specifies the TIS category:
# 2: 5UTR_ATG, 3: CDS_ATG, 4: 5UTR_nonATG, 5: CDS_nonATG

# Output Directory: The generated codon usage files will be stored in the `TIS_codon_library/` folder.
```

### Generate Features
```bash
python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 2
python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 3
python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 4
python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 5

# -TIS_code specifies the TIS category:
# 2: 5UTR_ATG, 3: CDS_ATG, 4: 5UTR_nonATG, 5: CDS_nonATG

# Output Directory: The generated feature files will be stored in the `feature/` folder.
```

### Train Model
```bash
python script/train_model.py -feature Features_Sl_CDSATG.txt
python script/train_model.py -feature Features_Sl_5UTRATG.txt
python script/train_model.py -feature Features_Sl_CDSnonATG.txt
python script/train_model.py -feature Features_Sl_5UTRnonATG.txt

# Output Directory: Trained models and associated files are stored in subdirectories named `Model_[feature name]` within the `model/` folder. The outputs include:
#
# - `Feature_scaler.pkl` and `Feature_scaler_attributes.tsv`: Feature scaling factors in PKL and TSV formats, respectively.
# - `Important_features.tsv`: Contains the importance of each feature.
# - `[classifier]_model.pkl`: Model file for the specified classifier (e.g., GradientBoosting, LogisticRegression, RandomForest, SVM) in PKL format.
# - `[classifier]_model_importances.tsv`: Importance of each feature for the corresponding classifier.
# - `Model_evaluation_results.tsv`: Evaluation results of the trained models.
```

### Cross Model Evaluation
```bash
python script/test_model.py -feature Features_Sl_5UTRATG.txt -model Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG

# Output Files: The evaluation results will be saved as `Cross_evaluation_result_[input feature name].tsv`.
```

### TIS Prediction
```bash
python script/predict_TIS.py -fasta input_data/predict_input.fa -models Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG

# Output Files: The prediction output for each FASTA sequence will be saved as `Predict_[FASTA header].tsv`.
# Each FASTA sequence generates one TSV file.
```

## Model Training

The model is trained using the following steps:
1. Feature extraction.
2. Train-test split.
3. Hyperparameter optimization using grid search or Bayesian optimization.
4. Model evaluation on a separate validation set.

Supported machine learning algorithms include:
- Random Forest (RF)
- Gradient Boosted Trees (GB)
- Support Vector Machines (SVM)
- Logistic Regression (LR)

## Evaluation

Evaluation metrics include:
- **F1**: Harmonic mean of precision and recall.
- **AUCROC**: Area under the Receiver Operating Characteristic curve.
- **MCC**: Matthews Correlation Coefficient for balanced evaluation of predictions.
- **Accuracy**: Overall prediction correctness.
- **Recall**: Proportion of actual TIS correctly identified.
- **Precision**: Proportion of true positive TIS predictions.
- **ConfusionMatrix**: Detailed classification performance representation.

Results are visualized using confusion matrices, ROC curves, and feature importance plots.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

