import sys  # nopep8
sys.path.append('.')  # nopep8

from core.utils.processingUtil import HLAProcessor

import typing
import os
import numpy as np
import pandas as pd

from collections import defaultdict
from numpy.random import RandomState, MT19937
from sklearn.model_selection import train_test_split, StratifiedKFold

# input
MSI011320_PATH = 'resources/datasets/raw_datasets/HLA_classI_MS_dataset_011320.tsv'
ANTI051821Z_PATH = 'resources/datasets/raw_datasets/antigen_information_051821_rev1.tsv'
ALLELE_MAPPER_PATH = 'resources/allele_mapper'
VOCAB_ALLELE_PATH = 'resources/VOCAB_ALLELE.txt'

# output
OUTPUT_MSI011320_PATH = 'resources/datasets/MSI011320'
OUTPUT_ANTI051821Z_PATH = 'resources/datasets/ANTI051821Z'
OUTPUT_MSI011320_ANTI051821Z_COMBINE_PATH = 'resources/datasets/MSI011320_ANTI051821Z_COMBINE'
OUTPUT_INTERMEDIET_PATH = 'resources/intermediate_netmhc2'

# parameter
MSI011320_KFOLD_NUM = 5
MSI011320_NEGATIVE_RATIO = 0.4
MSI011320_MT19937_SEEED = 3515
MSI011320_EVAL_SPLIT_SIZE = 0.25

if __name__ == '__main__':
    msi011320_df = pd.read_csv(MSI011320_PATH, sep='\t', index_col=False)
    msi011320_df_size_original = len(msi011320_df)
    # cleaning peptide (some row has allele name as peptide)
    msi011320_df = msi011320_df[np.logical_not(np.logical_or.reduce([msi011320_df['Peptide'].str.contains(unsual_amino, regex=False) for unsual_amino in "0-*:1235 746"]))]
    # cleaning allele name, there HLA-A*30:14L needs to be trimmed down to HLA-A*30:14
    msi011320_df.loc[:, 'Allele'] = msi011320_df['Allele'].str.slice(0, 11)  # length is capped at 11
    print(f"Cleaning MSI011320 dataset, resulted in {len(msi011320_df)} rows instead of {msi011320_df_size_original} rows")

    msi011320_pairs_set: typing.Set[typing.Tuple[str, str]] = set((msi011320_df[['Allele', 'Peptide']]).itertuples(index=False, name=None))

    anti051821z_df = pd.read_csv(ANTI051821Z_PATH, sep='\t', index_col=False)
    anti051821z_df.loc[:, 'Allele'] = anti051821z_df['HLA Allele'].apply(lambda x: f'HLA-{x[0]}*{x[1:3]}:{x[3:5]}')
    anti051821z_df.loc[:, 'has_binding_test'] = anti051821z_df['Binding Test'].notna()
    # for `isin_MSI011320` there is only one (the first row)
    anti051821z_df.loc[:, 'isin_MSI011320'] = [pair in msi011320_pairs_set for pair in (anti051821z_df[['Allele', 'Peptide']]).itertuples(index=False, name=None)]
    anti051821z_df.loc[:, 'is_outlier'] = anti051821z_df['Is Outlier'] == 'Yes'

    os.makedirs(OUTPUT_ANTI051821Z_PATH, exist_ok=True)
    anti051821z_df.to_csv(os.path.join(OUTPUT_ANTI051821Z_PATH, f"{os.path.splitext(os.path.basename(ANTI051821Z_PATH))[0]}_processed.csv"))  # index=False --> we need index
    # del anti051821z_df
    del msi011320_pairs_set
    print(f"Finished Prepare AITI051821Z")

    HLAProcessor.load_from_disk(VOCAB_ALLELE_PATH, ALLELE_MAPPER_PATH)

    # Compute peptide lenght distribution
    peptide_lenght_counter = defaultdict(int)  # type: typing.Dict[int, int]
    peptide: str
    for peptide in msi011320_df['Peptide']:
        peptide_lenght_counter[len(peptide)] += 1
    peptide_lenght_dist_dict: typing.Dict[int, float] = {lenght: peptide_lenght_counter[lenght] /
                                                         np.sum(list(peptide_lenght_counter.values())) for lenght, count in peptide_lenght_counter.items()}
    peptide_lenght_dist = sorted(list(peptide_lenght_dist_dict.items()), key=lambda x: x[0])
    del peptide_lenght_counter

    # Compute amino distribution and NEGATIVE_AMINO used for generating negative set
    # It is under commented to reduce computation time, just use the finished result
    # peptide_char_counter = defaultdict(int)  # type: typing.Dict[str, int]
    # for peptide in msi011320_df['Peptide']:
    #     for amino in peptide:
    #         peptide_char_counter[amino] += 1

    # peptide_char_dist_dict: typing.Dict[str, float] = {amino: peptide_char_counter[amino] /
    #                                                    np.sum(list(peptide_char_counter.values())) for amino, count in peptide_char_counter.items()}
    # peptide_char_dist = sorted(list(peptide_char_dist_dict.items()), key=lambda x: x[1], reverse=True)
    # del peptide_char_dist_dict
    # NEGATIVE_AMINO = [amino for amino, prop in peptide_char_dist if amino.isupper()][:20]
    NEGATIVE_AMINO = ['L', 'V', 'A', 'S', 'E', 'P', 'R', 'I', 'T', 'G', 'K', 'F', 'Y', 'Q', 'D', 'N', 'H', 'M', 'W', 'C']  # ['X', 'B'] were excluded
    print(f"Using hard coded NEGATIVE_AMINO={NEGATIVE_AMINO}")

    # Generate Negative for testing MSI011320
    rng = RandomState(seed=MT19937(MSI011320_MT19937_SEEED))  # type: np.random.mtrand.RandomState
    negative_length = int(len(msi011320_df)*MSI011320_NEGATIVE_RATIO)
    negative_protein_length_list = rng.choice([length for length, _ in peptide_lenght_dist], negative_length, p=[prob for _, prob in peptide_lenght_dist])
    negative_protein = ["".join(rng.choice(NEGATIVE_AMINO, length)) for length in negative_protein_length_list]  # I was using np.random here, too bad the generated peptide will be diff from mine
    neg_allele_index = rng.choice(msi011320_df.index, negative_length)
    msi011320_df.loc[:, 'isGenerated'] = np.repeat(False, len(msi011320_df))
    merged_msi011320_df = msi011320_df.append(pd.DataFrame({
        'Allele': msi011320_df.loc[neg_allele_index, 'Allele'],
        'Peptide': negative_protein,
        'isGenerated': np.repeat(True, negative_length),
    }), ignore_index=True)
    merged_msi011320_df.loc[:, 'isKnownAllele'] = merged_msi011320_df['Allele'].isin(HLAProcessor.mapper.keys())
    os.makedirs(OUTPUT_MSI011320_PATH, exist_ok=True)
    merged_msi011320_df.to_csv(os.path.join(OUTPUT_MSI011320_PATH, f"{os.path.splitext(os.path.basename(MSI011320_PATH))[0]}_processed-merged.csv"))

    # cross validation split
    rng = RandomState(seed=MT19937(MSI011320_MT19937_SEEED))  # so we can always reproduce k-fold
    cvKfold = StratifiedKFold(n_splits=MSI011320_KFOLD_NUM, shuffle=True, random_state=rng)
    train_index: np.ndarray
    test_index: np.ndarray
    for k, (train_index, test_index) in enumerate(cvKfold.split(merged_msi011320_df, merged_msi011320_df['isGenerated'])):
        kfoldRng = RandomState(seed=MT19937(MSI011320_MT19937_SEEED + (10*(k+1))))
        train_index, eval_index = train_test_split(train_index, test_size=MSI011320_EVAL_SPLIT_SIZE, random_state=kfoldRng)
        train_df = merged_msi011320_df.loc[train_index]
        eval_df = merged_msi011320_df.loc[eval_index]
        test_df = merged_msi011320_df.loc[test_index]
        train_df.to_csv(os.path.join(OUTPUT_MSI011320_PATH, f"{os.path.splitext(os.path.basename(MSI011320_PATH))[0]}_processed_kf-{k+1}_train.csv"))
        eval_df.to_csv(os.path.join(OUTPUT_MSI011320_PATH, f"{os.path.splitext(os.path.basename(MSI011320_PATH))[0]}_processed_kf-{k+1}_eval.csv"))
        test_df.to_csv(os.path.join(OUTPUT_MSI011320_PATH, f"{os.path.splitext(os.path.basename(MSI011320_PATH))[0]}_processed_kf-{k+1}_test.csv"))

    os.makedirs(OUTPUT_INTERMEDIET_PATH, exist_ok=True)
    with open(os.path.join(OUTPUT_INTERMEDIET_PATH, 'peptide_lenght_dist.yaml'), 'w') as fileHandler:
        for lenght, prob in peptide_lenght_dist:
            fileHandler.write(f"{lenght}: {prob}\n")
    with open(os.path.join(OUTPUT_INTERMEDIET_PATH, 'NEGATIVE_AMINO.txt'), 'w') as fileHandler:
        fileHandler.write(','.join(NEGATIVE_AMINO) + '\n')
    print("Finished prepare MSI011320")

    # combine the dataset for the second experiment
    os.makedirs(OUTPUT_MSI011320_ANTI051821Z_COMBINE_PATH, exist_ok=True)
    anti051821z_df_temp_selector = np.logical_and.reduce(
        [anti051821z_df.isin_MSI011320 == False, anti051821z_df.is_outlier == False, anti051821z_df.has_binding_test == False])
    anti051821z_df.loc[:, 'isGenerated'] = False
    anti051821z_df.loc[:, 'isKnownAllele'] = anti051821z_df['Allele'].isin(HLAProcessor.mapper.keys())
    combine_rng = RandomState(seed=MT19937(MSI011320_MT19937_SEEED))
    anti051821z_df_for_combine = anti051821z_df[anti051821z_df_temp_selector].sample(frac=1, random_state=combine_rng).reset_index(drop=True)
    msi011320_anti051821Z_combine_train_df = merged_msi011320_df.append(anti051821z_df[anti051821z_df_temp_selector], ignore_index=True).sample(frac=1, random_state=rng)
    # resplit it again, this time, append the anti
    rng = RandomState(seed=MT19937(MSI011320_MT19937_SEEED))
    cvKfold = StratifiedKFold(n_splits=MSI011320_KFOLD_NUM, shuffle=True, random_state=rng)
    train_index: np.ndarray
    test_index: np.ndarray
    anti_msi_combine_train: pd.DataFrame
    for k, ((train_index, test_index), anti_msi_combine_train) in enumerate(zip(cvKfold.split(merged_msi011320_df, merged_msi011320_df['isGenerated']), np.array_split(msi011320_anti051821Z_combine_train_df, MSI011320_KFOLD_NUM))):
        kfoldRng = RandomState(seed=MT19937(MSI011320_MT19937_SEEED + (10*(k+1))))
        train_index: list
        test_index: list
        train_index, eval_index = train_test_split(train_index, test_size=MSI011320_EVAL_SPLIT_SIZE, random_state=kfoldRng)
        train_combine, eval_combine = train_test_split(anti_msi_combine_train, test_size=MSI011320_EVAL_SPLIT_SIZE, random_state=kfoldRng)
        train_df = merged_msi011320_df.loc[train_index].append(train_combine, ignore_index=True).reset_index(drop=True)
        eval_df = merged_msi011320_df.loc[eval_index].append(eval_combine, ignore_index=True).reset_index(drop=True)
        test_df = merged_msi011320_df.loc[test_index]

        train_df.to_csv(os.path.join(OUTPUT_MSI011320_ANTI051821Z_COMBINE_PATH, f"{os.path.splitext(os.path.basename(MSI011320_PATH))[0]}_{os.path.splitext(os.path.basename(ANTI051821Z_PATH))[0]}_processed_kf-{k+1}_train.csv"))
        eval_df.to_csv(os.path.join(OUTPUT_MSI011320_ANTI051821Z_COMBINE_PATH, f"{os.path.splitext(os.path.basename(MSI011320_PATH))[0]}_{os.path.splitext(os.path.basename(ANTI051821Z_PATH))[0]}_processed_kf-{k+1}_eval.csv"))
        test_df.to_csv(os.path.join(OUTPUT_MSI011320_ANTI051821Z_COMBINE_PATH, f"{os.path.splitext(os.path.basename(MSI011320_PATH))[0]}_{os.path.splitext(os.path.basename(ANTI051821Z_PATH))[0]}_processed_kf-{k+1}_test.csv"))

    # your run may not be the same as mine as I load each test set and concat them and save here (due to I was using np.random.choice when I generate the peptide, but yours should've been fixed)
    msi011320_anti051821Z_combine_train_df.to_csv(os.path.join(OUTPUT_MSI011320_ANTI051821Z_COMBINE_PATH, f"{os.path.splitext(os.path.basename(MSI011320_PATH))[0]}_{os.path.splitext(os.path.basename(ANTI051821Z_PATH))[0]}_processed-merged.csv"))

    print("Preparation successfully completed")
