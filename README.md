# MHCSeqNet2

This is the official repository containing the code to reproduce result founded on a research paper titiled `MHCSeqNet2 - Improved Peptide-Class I MHC Binding Prediction for Alleles with Low Data`

## Index

- [How to prepare Environment](#how-to-prepare-environment)
- [How to Inference](#how-to-inference)
  -  [How to Reproduce Inference Result](#how-to-reproduce-inference-result)
  -  [How to Reproduce Figure](#how-to-reproduce-figure)
- [How to Train Prediction Model](#how-to-train-prediction-model)
- [How to Train Pre-Training Model](#how-to-train-pre-training-model)
  - [How to Train 3D Allele Pre-Training Model](#how-to-train-3d-allele-pre-training-model)
  - [How to obtain the embedding weight](#how-to-obtain-the-embedding-weight)
- [Data Preparation](#data-preparation)
  - [Prepare Pre-training Dataset](#prepare-pre-training-dataset)
  - [Prepare Predictor Dataset](#prepare-predictor-dataset)

## How to prepare Environment

1. Clone this repository

```bash
git clone https://github.com/cmb-chula/MHCSeqNet2.git
```

2. Edit docker-compose.yaml to change the volume mount location to suit your case.

3. Use this command to create container

```bash
docker-compose up -d --build
```

4. Then you are free to access the container using exec

```bash
docker exec -it zenthesisv2-dev-mhcseqnet2-1 bash
```

## How to Inference

1. First you must obtain/[train](#how-to-train-prediction-model) the prediction model weight

```bash
mkdir -p resources/trained_weight/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/final_model.tar.gz -O - | tar -xz -C resources/trained_weight/
```

2. You can use file `mhctool.py` to view the usage and available options.

```
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
But to reproduce the result, one could use the following commands.

```code
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-1_test.csv" \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-1_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-1_test_raw.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-2_test.csv" \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-2_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-2_test_raw.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-3_test.csv" \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-3_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-3_test_raw.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-4_test.csv" \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-4_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-4_test_raw.csv"
python mhctool.py \
    --MODE CSV \
    --CSV_PATH "resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-5_test.csv" \
    --PEPTIDE_COLUMN_NAME Peptide \
    --ALLELE_COLUMN_NAME Allele \
    --GPU_ID 0 \
    --ALLELE_MAPPER_PATH resources/allele_mapper \
    --OUTPUT_DIRECTORY "/tmp/prediction_result/HLA_classI_MS_dataset_011320_processed_kf-5_test_raw.csv" \
    --TEMP_FILE_PATH "/tmp/prediction_result/_tmp_HLA_classI_MS_dataset_011320_processed_kf-5_test_raw.csv"
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

```bash
python scripts/make_figure_auc_full_vs_few_zoom.py
```

## How to Train Prediction Model

1. Visit [Data Preparation](#data-preparation)
2. Obtain [pretrain weght](#how-to-obtain-the-embedding-weight) or train the [pre-train model](#how-to-train-pre-training-model)
2. Run the following commands to start training

  Please note that each fold can be trained simultaneously

```bash
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

## How to Train Pre-Training Model

For how to train pllele pre-training model, stay tuned!  
For now, you could obtain the pre-train embedding from release

### How to Train 3D Allele Pre-Training Model

1. Visit [Data Preparation](#data-preparation)
2. Train with the command below

```bash
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

```bash
# python scripts/extract_embedding.py
echo "After the training is completed, central and context embedding weight will be available in the saved model folder"
```

### How to obtain the embedding weight

```bash
mkdir -p resources/intermediate_netmhc2/
mkdir -p resources/trained_weight/embedding-3d/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/peptide_central_embedding.tar.gz -O - | tar -xz -C resources/intermediate_netmhc2/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/central_embeddings_matrix.tar.gz -O - | tar -xz -C resources/trained_weight/embedding-3d/
```

## Data Preparation

### Prepare Pre-training Dataset

1. Obtain raw 3D allele and peptide dataset from release page

```bash
mkdir -p resources/datasets/PRETRAIN_HUMAN_PROTEIN/
mkdir -p resources/datasets/PRETRAIN_3D/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/humanProtein_peptide.tar.gz -O - | tar -xz -C resources/datasets/PRETRAIN_HUMAN_PROTEIN/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/raw_3d_dataset.tar.gz -O - | tar -xz -C resources/datasets/PRETRAIN_3D/
```

2. Run prepare script to create dataset

```bash
python scripts/prepare_pretraining_human_protien.py
python scripts/prepare_pretraining_3d_allele.py
```

### Prepare Predictor Dataset

1. Obtain dataset from release page

```bash
mkdir -p resources/datasets/raw_datasets/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/HLA_classI_MS_dataset_011320.tar.gz -O - | tar -xz -C resources/datasets/raw_datasets/
wget -c https://github.com/cmb-chula/MHCSeqNet2/releases/download/v1.0/antigen_information_051821_rev1.tar.gz -O - | tar -xz -C resources/datasets/raw_datasets/
```

2. Run prepare script to create dataset

```bash
python scripts/prepare.py
```
