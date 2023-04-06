"""
Prepare pretraining
Estentially there are two things that's needed to train GloVe with FastText style.

1. central2context --> dict mapping central word to its context word
2. pair_map_distribution --> how do we sample the central word

So this estentially convert human peptide into those two things 

To use this just create pair_map_distribution from saved pair_map_counter

    word_map_couter = {pair: count for pair, count in pair_map_counter.items() if len(pair[0]) == len(pair[1]) == WORD_LENGHT}
    pair_map_distribution = np.asarray(list(word_map_couter.values())) # check your data sorting
    pair_map_distribution = pair_map_distribution/np.sum(pair_map_distribution)
"""

import sys  # nopep8
sys.path.append('.')  # nopep8

from typing import Dict, List, Set, Tuple
from tqdm.auto import tqdm
from collections import defaultdict
from multiprocessing import Pool
from core.utils.processingUtil import HLAProcessor, PeptideGenerator

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# input
PRETRAIN_HUMAN_PROTEIN_PATH = "resources/datasets/PRETRAIN_HUMAN_PROTEIN/humanProtein_peptide.txt"
VOCAB_PATH = 'resources/VOCAB.txt'
peptide_lenght_dist_path = 'resources/intermediate_netmhc2/peptide_lenght_dist.yaml'
NEGATIVE_AMINO_PATH = 'resources/NEGATIVE_AMINO.txt'

# output
OUTPUT_PATH = "resources/datasets/PRETRAIN_HUMAN_PROTEIN/"

# parameters
EPSILON = 1e-4
WINDOW_SIZE = 3
WORD_LENGHT = 3  # characters


def process_seq(human_protein_df_partitioned: pd.DataFrame,
                processnumber: int,
                word_lenght: int = WORD_LENGHT):
    global AMINO_ACIDS_WITH_UNKNOWN_SET
    print(f"process_seq({len(human_protein_df_partitioned)}), at process number {processnumber}")
    pair_map: Dict[Tuple[str, str], int] = defaultdict(int)
    seq: str
    for idx, seq in human_protein_df_partitioned['Sequence'].items():
        seq = '^' + seq
        seq_len = len(seq)
        n_words = [seq[i: i + word_lenght] for i in range(seq_len - word_lenght + 1)]
        for i in range(len(n_words)):
            for word_near in n_words[max(0, i-WINDOW_SIZE):i] + n_words[i+1:min(seq_len, i+WINDOW_SIZE+1)]:
                if(np.any([amino not in AMINO_ACIDS_WITH_UNKNOWN_SET for amino in list(word_near)])):
                    continue
                pair_map[(n_words[i], word_near)] += 1
    return pair_map


if __name__ == "__main__":
    # HLAProcessor.load_from_disk(VOCAB_ALLELE_PATH, ALLELE_MAPPER_PATH)
    PeptideGenerator.load_from_disk(VOCAB_PATH, peptide_lenght_dist_path, NEGATIVE_AMINO_PATH)
    AMINO_ACIDS_WITH_UNKNOWN_SET = set(PeptideGenerator.PEPTIDE_LEN_ONE_MAPPER_LIST)

    human_protein_df: pd.DataFrame = pd.read_csv(PRETRAIN_HUMAN_PROTEIN_PATH, sep='\t')

    pair_map_counter: Dict[Tuple[str, str], int] = defaultdict(int)  # For making distribution
    NUM_PROCESS = 30
    with Pool(processes=NUM_PROCESS+1) as pool:
        with tqdm(total=NUM_PROCESS+1, desc='multi processing') as pbar:
            parition_count = (len(human_protein_df)//NUM_PROCESS)
            starmap_arg = zip(list(
                [human_protein_df.iloc[i:i+parition_count] for i in range(0, len(human_protein_df), parition_count)]),
                range(NUM_PROCESS+1),
                [WORD_LENGHT]*(NUM_PROCESS+1))
            for pair_map_counter_child in (pool.starmap(process_seq, starmap_arg)):
                # need to merge each dict
                for pair, count in pair_map_counter_child.items():
                    pair_map_counter[pair] += count
                pbar.update(1)

    human_protein_central_to_context: Dict[str, List[str]] = defaultdict(list)
    for central, context in pair_map_counter:
        human_protein_central_to_context[central].append(context)

    output_base_path = os.path.join(OUTPUT_PATH, f"human-protein_word-size-{WORD_LENGHT}_window-size-{WINDOW_SIZE}")
    output_central2context_path = os.path.join(output_base_path, f"central2context.yaml")
    output_pair_map_counter_path = os.path.join(output_base_path, f"pair_map_counter.yaml")

    os.makedirs(output_base_path, exist_ok=True)

    # simplify data for saving
    saving_central2context = {central: list(context) for central, context in human_protein_central_to_context.items()}
    saving_pair_map_counter = {'counter': [{'pair': list(pair), 'count': count} for pair, count in pair_map_counter.items()]}

    # dumping pkl is an option but it not convenient to know what's going on in this file
    with open(output_central2context_path, 'w') as fileHandler:
        yaml.safe_dump(saving_central2context, fileHandler, indent=4)
    with open(output_pair_map_counter_path, 'w') as fileHandler:
        yaml.safe_dump(saving_pair_map_counter, fileHandler, indent=4)
