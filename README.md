# MHCSeqNet2

This is the official repository containing the code to reproduce result founded on a research paper titled `MHCSeqNet2 - Improved Peptide-Class I MHC Binding Prediction for Alleles with Low Data`

## Index

- [How to prepare Environment](#how-to-prepare-environment)
- [How to Inference](#how-to-inference)
  - [How to Reproduce Inference Result](#how-to-reproduce-inference-result)
    - [Reproduce Result from Model with Publicly Available Dataset](#reproduce-result-from-model-with-publicly-available-dataset)
    - [Reproduce Result from Model with SMSNet Dataset](#reproduce-result-from-model-with-smsnet-dataset)
  - [How to Reproduce Figure](#how-to-reproduce-figure)
- [How to Train Prediction Model](#how-to-train-prediction-model)
- [How to Train Pre-Training Model](#how-to-train-pre-training-model)
  - [How to Train 3D Allele Pre-Training Model](#how-to-train-3d-allele-pre-training-model)
  - [How to obtain the embedding weight](#how-to-obtain-the-embedding-weight)
- [Data Preparation](#data-preparation)
  - [Prepare Pre-training Dataset](#prepare-pre-training-dataset)
  - [Prepare Predictor Dataset](#prepare-predictor-dataset)
- [Dataset References](#dataset-references)

## Quick Message from the owner

To avoid confusion, I accidentally set the training label in reverse order.  
This resulted in prediction `0` means `bind` while `1` means `not bind`.  
And `isGenerated` Column is reversed order as well.  
I want to express my sincere apology here, if there is a new version and the issue has been resolved, I'll announce it.  

## How to prepare Environment

1. Clone this repository

```shell
git clone https://github.com/cmb-chula/MHCSeqNet2.git
```

2. Edit docker-compose.yaml to change the volume mount location to suit your case.

3. Use this command to create container

```shell
docker-compose up -d --build
```

4. Then you are free to access the container using exec

```shell
docker exec -it mhcseqnet2_dev-mhcseqnet2_1 bash
```

## How to Inference

1. First you must obtain/[train](#how-to-train-prediction-model) the prediction model weight

```shell
mkdir -p resources/trained_weight/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/final_model.tar.gz -O - | tar -xz -C resources/trained_weight/
# By default mhctool.py will uses final_model_with_smsnetdata as its weight
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/final_model_with_smsnetdata.tar.gz -O - | tar -xz -C resources/trained_weight/
```

2. You can use file `mhctool.py` to view the usage and available options\*\*.

  \*\*For real application it's recommended to always include `--USE_ENSEMBLE` flag\*\*

```shell
$ python mhctool.py --help
usage: mhctool.py [-h] [--MODE {CSV,CROSS}] [--CSV_PATH CSV_PATH] [--PEPTIDE_COLUMN_NAME PEPTIDE_COLUMN_NAME] [--ALLELE_COLUMN_NAME ALLELE_COLUMN_NAME] [--PEPTIDE_PATH PEPTIDE_PATH]
                  [--ALLELE_PATH ALLELE_PATH] [--IGNORE_UNKNOW] [--LOG_UNKNOW] [--LOG_UNKNOW_PATH LOG_UNKNOW_PATH] [--GPU_ID GPU_ID] [--USE_ENSEMBLE]
                  [--MODEL_TYPE {MHCSeqNet2,MHCSeqNet2_GRUPeptide,GloVeFastText,MultiHeadGloVeFastTextSplit,MultiHeadGloVeFastTextJointed}] [--ALLELE_MAPPER_PATH ALLELE_MAPPER_PATH]
                  [--OUTPUT_DIRECTORY OUTPUT_DIRECTORY] [--TEMP_FILE_PATH TEMP_FILE_PATH] [--SUPPRESS_LOG]

MHCTool

optional arguments:
  -h, --help            show this help message and exit
  --MODE {CSV,CROSS}    Mode `CSV` or `CROSS` Select the mode to run, the tool will execute based on current selection • csv mode allow you to choose a csv/tsv file which must contain the column
                        for peptide and allele • cross mode allow you to choose two files one containing peptides, and the other containing alleles which will be crossed together
  --CSV_PATH CSV_PATH   path directory to input csv when use `--MODE CSV`
  --PEPTIDE_COLUMN_NAME PEPTIDE_COLUMN_NAME
                        the column name which containing the peptide
  --ALLELE_COLUMN_NAME ALLELE_COLUMN_NAME
                        the column name which containing the allele
  --PEPTIDE_PATH PEPTIDE_PATH
                        path directory to input peptide when use `--MODE CROSS`
  --ALLELE_PATH ALLELE_PATH
                        path directory to input allele when use `--MODE CROSS`
  --IGNORE_UNKNOW       if setted it will skip the unknown
  --LOG_UNKNOW          if setted it will log the unknown that was skipped
  --LOG_UNKNOW_PATH LOG_UNKNOW_PATH
                        the file which the unknow will be logged to
  --MODEL_KF MODEL_KF   specify model weight to use, if not using ensemble
  --GPU_ID GPU_ID       default GPU, you can specify a GPU to be used by given a number i.e, `--GPU_ID 0`
  --USE_ENSEMBLE        Run the result multiple times on multiple models and use the average as the score
  --MODEL_TYPE {MHCSeqNet2,MHCSeqNet2_GRUPeptide,GloVeFastText,MultiHeadGloVeFastTextSplit,MultiHeadGloVeFastTextJointed}
                        specify model to use
  --ALLELE_MAPPER_PATH ALLELE_MAPPER_PATH
                        path to the folder that contain yaml file needed for the tool. You can use this to add a new allele, please visit readme for more
  --OUTPUT_DIRECTORY OUTPUT_DIRECTORY
                        where to save the final result to (only .csv or .tsv)
  --TEMP_FILE_PATH TEMP_FILE_PATH
                        path to intermediate result file to maintain system compatibility and stability, the program need to store intermediate result.
  --SUPPRESS_LOG        use to suppress log, useful only for running from gui
```

### How to Reproduce Inference Result

Normally, after training has completed, `train.py` will predict the result of each CV on its test set.  
But to reproduce the result, one could use the following steps.

#### Reproduce Result from Model with Publicly Available Dataset

1. Edit model weight to limit to publicly available data at file [`mhctool.py`](https://github.com/cmb-chula/MHCSeqNet2/blob/0029715a78bc2336ac19b1f731c365daaa40fb42/mhctool.py#L69-L82)

2. Use the following commands

```code
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-1_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 0 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-1_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-1_test_raw.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-2_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 1 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-2_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-2_test_raw.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-3_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 2 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-3_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-3_test_raw.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-4_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 3 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-4_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-4_test_raw.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-5_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 4 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-5_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-5_test_raw.csv"
```

#### Reproduce Result from Model with SMSNet Dataset

1. Edit model weight to limit to SMSNet data (If you hadn't edited anything yet, there's nothing to change) at file [`mhctool.py`](https://github.com/cmb-chula/MHCSeqNet2/blob/0029715a78bc2336ac19b1f731c365daaa40fb42/mhctool.py#L69-L82)

2. Use the following commands

```code
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320_ANTI051821Z_COMBINE/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-1_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 0 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-1_test.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-1_test.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320_ANTI051821Z_COMBINE/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-2_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 1 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-2_test.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-2_test.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320_ANTI051821Z_COMBINE/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-3_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 2 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-3_test.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-3_test.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320_ANTI051821Z_COMBINE/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-4_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 3 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-4_test.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-4_test.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320_ANTI051821Z_COMBINE/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-5_test.csv" \
    --IGNORE_UNKNOW \
    --MODEL_KF 4 \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-5_test.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-5_test.csv"
```

### How to Reproduce Figure

1. [Obtain](#how-to-inference)/[train](#how-to-train-prediction-model) predictor model

2. Edit `scripts/make_figure_auc_full_vs_few_zoom.py` file in section `KFOLD_RESULT_PATH` to match with your model path

```python
KFOLD_RESULT_PATH: typing.List[typing.Tuple[str, str, str, str, bool, bool, str]] = [
    ('ExperimentalResult', 'this work', 'Prediction', 'isGenerated', True, True, 'resources/trained_weight/final_model'),
]
```

3. Run make figure script

```shell
python scripts/make_figure_auc_full_vs_few_zoom.py
```

## How to Train Prediction Model

1. Visit [Data Preparation](#data-preparation)
2. Obtain [pre-train weight](#how-to-obtain-the-embedding-weight) or train the [pre-train model](#how-to-train-pre-training-model)
2. Run the following commands to start training

  Please note that each fold can be trained simultaneously

```shell
python train.py \
    --dataset=MSI011320 \
    --root_dir=resources/datasets \
    --run_kfold 1 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model \
    --experiment_name=final_model \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256

python train.py \
    --dataset=MSI011320 \
    --root_dir=resources/datasets \
    --run_kfold 2 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model \
    --experiment_name=final_model \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256

python train.py \
    --dataset=MSI011320 \
    --root_dir=resources/datasets \
    --run_kfold 3 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model \
    --experiment_name=final_model \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256

python train.py \
    --dataset=MSI011320 \
    --root_dir=resources/datasets \
    --run_kfold 4 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model \
    --experiment_name=final_model \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256

python train.py \
    --dataset=MSI011320 \
    --root_dir=resources/datasets \
    --run_kfold 5 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model \
    --experiment_name=final_model \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256
```

3. Or train model with SMSNet data using the following commands

```shell
python train.py \
    --dataset=MSI011320_ANTI051821Z_COMBINE \
    --root_dir=resources/datasets \
    --run_kfold 1 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model_with_smsnetdata \
    --experiment_name=final_model_with_smsnetdata \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256
python train.py \
    --dataset=MSI011320_ANTI051821Z_COMBINE \
    --root_dir=resources/datasets \
    --run_kfold 2 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model_with_smsnetdata \
    --experiment_name=final_model_with_smsnetdata \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256
python train.py \
    --dataset=MSI011320_ANTI051821Z_COMBINE \
    --root_dir=resources/datasets \
    --run_kfold 3 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model_with_smsnetdata \
    --experiment_name=final_model_with_smsnetdata \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256
python train.py \
    --dataset=MSI011320_ANTI051821Z_COMBINE \
    --root_dir=resources/datasets \
    --run_kfold 4 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model_with_smsnetdata \
    --experiment_name=final_model_with_smsnetdata \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256
python train.py \
    --dataset=MSI011320_ANTI051821Z_COMBINE \
    --root_dir=resources/datasets \
    --run_kfold 5 \
    --load_embedding_peptide \
    --load_embedding_allele \
    --embedding_allele_path=resources/trained_weight/embedding-3d/central_embeddings_matrix.npy \
    --save_path=resources/trained_weight/final_model_with_smsnetdata \
    --experiment_name=final_model_with_smsnetdata \
    --epoch 420 \
    --early_stop_patience 150 \
    --batch_size_train=256 \
    --batch_size_test=256
```

## How to Train Pre-Training Model

For how to train peptide pre-training model, stay tuned!  
For now, you could obtain the pre-train embedding from release

### How to Train 3D Allele Pre-Training Model

1. Visit [Data Preparation](#data-preparation)
2. Train with the command below

```shell
python train.py \
    --MODEL_TYPE=GloVeFastText \
    --dataset=PRETRAIN_3D \
    --save_path=resources/trained_weight/ \
    --experiment_name=embedding-3d \
    --central2context_path="resources/datasets/PRETRAIN_3D/dist-avg-distance_threshold_45/central2context.yaml" \
    --pair_map_counter_path="resources/datasets/PRETRAIN_3D/dist-avg-distance_threshold_45/pair_map_counter.yaml" \
    --batch_size_train=256 \
    --epoch=50 \
    --checkpoint_monitor='acc' \
    --reduce_lr_monitor='loss' \
    --reduce_lr_patience=2 \
    --early_stop_monitor='acc' \
    --early_stop_patience=3
```

3. ~~Set the model weight path inside `scripts/extract_embedding.py` to match with your path~~

4. ~~Extract the embedding weight~~

```shell
# python scripts/extract_embedding.py
echo "After the training is completed, central and context embedding weight will be available in the saved model folder"
```

### How to obtain the embedding weight

```shell
mkdir -p resources/intermediate_netmhc2/
mkdir -p resources/trained_weight/embedding-3d/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/peptide_central_embedding.tar.gz -O - | tar -xz -C resources/intermediate_netmhc2/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/central_embeddings_matrix.tar.gz -O - | tar -xz -C resources/trained_weight/embedding-3d/
```

## Data Preparation

### Prepare Pre-training Dataset

1. Obtain raw 3D allele and peptide dataset from release page

```shell
mkdir -p resources/datasets/PRETRAIN_HUMAN_PROTEIN/
mkdir -p resources/datasets/PRETRAIN_3D/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/humanProtein_peptide.tar.gz -O - | tar -xz -C resources/datasets/PRETRAIN_HUMAN_PROTEIN/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/raw_3d_dataset.tar.gz -O - | tar -xz -C resources/datasets/PRETRAIN_3D/
```

2. Run prepare script to create dataset

```shell
python scripts/prepare_pretraining_human_protein.py
python scripts/prepare_pretraining_3d_allele.py
```

### Prepare Predictor Dataset

The first HLA binding dataset (`HLA_classI_MS_dataset_011320`) comes from combining several mass spectrometry-based mono-allelic HLA peptidomics studies [[1]](#1)[[2]](#2)[[3]](#3)[[4]](#4)[[5]](#5) with peptide-HLA pairs curated by the Immune Epitope Database (IEDB[[6]](#6)). Duplicated peptide-HLA pairs and peptides with modifications were removed. In total, there were 514,928 peptide-HLA pairs across 164 alleles

The second HLA binding dataset (`antigen_information_051821_rev1`) was
derived by applying SMSNet, a de novo peptide sequencing
tool, to re-analyze two large mono-allelic HLA peptidomics
datasets [[3]](#3)[[4]](#4). This new dataset was recently explored [[7]](#7)
but has not yet been utilized for HLA binding prediction. In
total, 43,190 new peptide-HLA pairs across 89 alleles with
peptide lengths within 8-15 amino acids were identified.

1. Obtain dataset from release page

```shell
mkdir -p resources/datasets/raw_datasets/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/HLA_classI_MS_dataset_011320.tar.gz -O - | tar -xz -C resources/datasets/raw_datasets/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/antigen_information_051821_rev1.tar.gz -O - | tar -xz -C resources/datasets/raw_datasets/
```

2. Run prepare script to create dataset

```shell
python scripts/prepare.py
```

## Dataset References
<a id="1">[1]</a> 
M. Di Marco, H. Schuster, L. Backert, M. Ghosh, H.-G. Rammensee,
and S. Stevanovi ́c, “Unveiling the Peptide Motifs of HLA-C and HLA-G
from Naturally Presented Peptides and Generation of Binding Prediction
Matrices,” J Immunol, vol. 199, DOI 10.4049/jimmunol.1700938, no. 8,
pp. 2639–2651, Sep. 2017. [Online]. Available: [https://doi.org/10.4049/jimmunol.1700938](https://doi.org/10.4049/jimmunol.1700938)

<a id="2">[2]</a> 
M. Solleder, P. Guillaume, J. Racle, J. Michaux, H.-S. Pak, M. Müller,
G. Coukos, M. Bassani-Sternberg, and D. Gfeller, “Mass Spectrometry
Based Immunopeptidomics Leads to Robust Predictions of Phospho-
rylated HLA Class I Ligands,” Mol Cell Proteomics, vol. 19, DOI
10.1074/mcp.TIR119.001641, no. 2, pp. 390–404, Dec. 2019. [Online].
Available: [https://doi.org/10.1074/mcp.TIR119.001641](https://doi.org/10.1074/mcp.TIR119.001641)

<a id="3">[3]</a> 
J. G. Abelin, D. B. Keskin, S. Sarkizova, C. R. Hartigan, W. Zhang,
J. Sidney, J. Stevens, W. Lane, G. L. Zhang, T. M. Eisenhaure, K. R.
Clauser, N. Hacohen, M. S. Rooney, S. A. Carr, and C. J. Wu, “Mass
Spectrometry Profiling of HLA-Associated Peptidomes in Mono-allelic
Cells Enables More Accurate Epitope Prediction,” Immunity, vol. 46,
DOI https://doi.org/10.1016/j.immuni.2017.02.007, no. 2, pp. 315–326,
Feb. 2017. [Online]. Available: [https://www.sciencedirect.com/science/article/pii/S1074761317300420](https://www.sciencedirect.com/science/article/pii/S1074761317300420)

<a id="4">[4]</a> 
S. Sarkizova, S. Klaeger, P. M. Le, L. W. Li, G. Oliveira, H. Keshishian,
C. R. Hartigan, W. Zhang, D. A. Braun, K. L. Ligon, P. Bachireddy, I. K.
Zervantonakis, J. M. Rosenbluth, T. Ouspenskaia, T. Law, S. Justesen,
J. Stevens, W. J. Lane, T. Eisenhaure, G. Lan Zhang, K. R. Clauser,
N. Hacohen, S. A. Carr, C. J. Wu, and D. B. Keskin, “A large
peptidome dataset improves HLA class I epitope prediction across
most of the human population,” Nature Biotechnology, vol. 38, DOI
10.1038/s41587-019-0322-9, no. 2, pp. 199–209, Feb. 2020. [Online].
Available: [https://doi.org/10.1038/s41587-019-0322-9](https://doi.org/10.1038/s41587-019-0322-9)

<a id="5">[5]</a> 
J. G. Abelin, D. Harjanto, M. Malloy, P. Suri, T. Colson, S. P.
Goulding, A. L. Creech, L. R. Serrano, G. Nasir, Y. Nasrul-
lah, C. D. McGann, D. Velez, Y. S. Ting, A. Poran, D. A.
Rothenberg, S. Chhangawala, A. Rubinsteyn, J. Hammerbacher,
R. B. Gaynor, E. F. Fritsch, J. Greshock, R. C. Oslund,
D. Barthelme, T. A. Addona, C. M. Arieta, and M. S. Rooney,
“Defining HLA-II Ligand Processing and Binding Rules with
Mass Spectrometry Enhances Cancer Epitope Prediction,” Immunity,
vol. 51, DOI 10.1016/j.immuni.2019.08.012, no. 4, pp. 766–779.e17,
2019. [Online]. Available: [https://www.sciencedirect.com/science/article/pii/S1074761319303632](https://www.sciencedirect.com/science/article/pii/S1074761319303632)

<a id="6">[6]</a> 
R. Vita, S. Mahajan, J. A. Overton, S. K. Dhanda, S. Martini, J. R.
Cantrell, D. K. Wheeler, A. Sette, and B. Peters, “The Immune Epitope
Database (IEDB): 2018 update,” Nucleic Acids Research, vol. 47, DOI
10.1093/nar/gky1006, no. D1, pp. D339–D343, 10 2018. [Online].
Available: [https://doi.org/10.1093/nar/gky1006](https://doi.org/10.1093/nar/gky1006)

<a id="7">[7]</a> 
B. Reynisson, B. Alvarez, S. Paul, B. Peters, and M. Nielsen,
“NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC
antigen presentation by concurrent motif deconvolution and integration
of MS MHC eluted ligand data,” Nucleic Acids Research, vol. 48, DOI
10.1093/nar/gkaa379, no. W1, pp. W449–W454, 05 2020. [Online].
Available: [https://doi.org/10.1093/nar/gkaa379](https://doi.org/10.1093/nar/gkaa379)