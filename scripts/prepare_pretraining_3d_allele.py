"""
Prepare pretraining
Estentially there are two things that's needed to train GloVe with FastText style.

1. central2context --> dict mapping central word to its context word
2. pair_map_distribution --> how do we sample the central word

So this estentially convert 3d average std dist matrix into those two things

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
from core.utils.processingUtil import HLAProcessor
from multiprocessing import Pool

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# input
PRETRAIN_3D_AVG_DISTANCE_PATH = "resources/datasets/PRETRAIN_3D/raw_3d_dataset/HLA_classI_3D_avg_dist.tsv"
# PRETRAIN_3D_STD_DISTANCE_PATH = "resources/datasets/PRETRAIN_3D/raw_3d_dataset/HLA_classI_3D_std_dist.tsv"
PRETRAIN_3D_ALIGNED_HLA_PATH = "resources/datasets/PRETRAIN_3D/raw_3d_dataset/HLA_classI_aligned_sequence.txt"
ALLELE_MAPPER_PATH = "resources/allele_mapper"
VOCAB_ALLELE_PATH = "resources/VOCAB_ALLELE.txt"

# output
OUTPUT_PATH = "resources/datasets/PRETRAIN_3D/"

# parameters
EPSILON = 1e-4
WINDOW_SIZE = 3
WORD_LENGHT = 3  # characters
# DIST_THRESHOLD = 0.3
# DIST_THRESHOLD_EXTRACT_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
DIST_THRESHOLD_EXTRACT_LIST = [45] # The final model uses threshold of 45


def process_seq(raw_mhc: List[Tuple[str, str]],
                processnumber: int,
                dist_mat: np.ndarray,
                value_region_dist_mat: Tuple[Tuple[int, int], Tuple[int, int]],
                dist_threshold: float = 0.3,):
    """
    This function take a look at the 3d distance matrix, and each HLA, then, use the given threshold to create central and context for fastText dataset
    """
    print(f"process_seq({len(raw_mhc)}), at process number {processnumber}")
    central2context: Dict[str, Set] = defaultdict(set)  # For making negative and positive sameple
    pair_map_counter: Dict[Tuple[str, str], int] = defaultdict(int)  # For making distribution
    _name: str
    seq: str
    for _name, seq in tqdm(raw_mhc):
        for i in range(value_region_dist_mat[0][0], value_region_dist_mat[1][0]):  # - WORD_LENGHT):
            for w in range(WORD_LENGHT):  # maxWordLength
                for contextidx in (np.arange(dist_mat[i].shape[0])[np.logical_and(dist_mat[i] > EPSILON, np.mean(dist_mat[i: i+w+1], axis=0) < dist_threshold)]):
                    central2context[seq[i: i+w+1]].add(seq[contextidx: contextidx+w+1])
                    pair_map_counter[(seq[i: i+w+1], seq[contextidx: contextidx+w+1])] += 1
    return central2context, pair_map_counter


if __name__ == "__main__":
    HLAProcessor.load_from_disk(VOCAB_ALLELE_PATH, ALLELE_MAPPER_PATH)

    for dist_threshold in DIST_THRESHOLD_EXTRACT_LIST:

        print(f"Preprocessing for threshold = {dist_threshold}")

        dist_mat = pd.read_csv(PRETRAIN_3D_AVG_DISTANCE_PATH, sep='\t', header=None).to_numpy()
        # Expected to be ((23, 23), (301, 301))
        value_region_dist_mat = ((np.argmax(dist_mat[dist_mat.shape[1]//2, :] > EPSILON), np.argmax(dist_mat[dist_mat.shape[0]//2, :] > EPSILON)),
                                 (dist_mat.shape[1] - np.argmax(dist_mat[dist_mat.shape[1]//2, :][::-1] > EPSILON), dist_mat.shape[0] - np.argmax(dist_mat[dist_mat.shape[0]//2, :][::-1] > EPSILON)))

        # # This is how I extract the image for the figure
        # cropped_dist_mat = dist_mat[value_region_dist_mat[0][0]:value_region_dist_mat[1][0], value_region_dist_mat[0][1]:value_region_dist_mat[1][1]]
        # fig = plt.figure(figsize=(10, 10,))
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # plt.set_cmap('jet')
        # ax.imshow(cropped_dist_mat, aspect='equal')
        # plt.savefig("figures/3d_dist_map.png", dpi=300)
        # breakpoint()

        with open(PRETRAIN_3D_ALIGNED_HLA_PATH, 'r') as fileHandler:
            used_hla = set()
            raw_mhc: List[Tuple[str, str]] = []
            for line in fileHandler.readlines():
                name, seq = line.strip().split('\t', 1)
                name = name[:7]
                if name in used_hla:
                    continue
                raw_mhc.append((name, seq))
                used_hla.add(name)

        # run the processing
        central2context: Dict[str, Set] = defaultdict(set)  # For making negative and positive sameple
        pair_map_counter: Dict[Tuple[str, str], int] = defaultdict(int)  # For making distribution
        NUM_PROCESS = 30
        with Pool(processes=NUM_PROCESS+1) as pool:
            with tqdm(total=NUM_PROCESS+1, desc='multi processing') as pbar:
                parition_count = (len(raw_mhc)//NUM_PROCESS)
                starmap_arg = zip(list(
                    [raw_mhc[i:i+parition_count] for i in range(0, len(raw_mhc), parition_count)]),
                    range(NUM_PROCESS+1),
                    [dist_mat]*(NUM_PROCESS+1),
                    [value_region_dist_mat]*(NUM_PROCESS+1),
                    [dist_threshold]*(NUM_PROCESS+1))
                for central2context_child, pair_map_counter_child in (pool.starmap(process_seq, starmap_arg)):
                    # need to merge each dict
                    for central, contexts in central2context_child.items():
                        central2context[central] = central2context[central].union(contexts)
                    for pair, count in pair_map_counter_child.items():
                        pair_map_counter[pair] += count
                    pbar.update(1)

        # sanity check, the vocab should not produce any new word
        vocab_mhc_list = []
        for central, contexts in central2context.items():
            vocab_mhc_list.append(central)
            vocab_mhc_list.extend(contexts)

        assert len(set(vocab_mhc_list).difference(set(HLAProcessor.protein2int_mhc.keys()))) == 0, "the central and context should not produce any new word"
        del vocab_mhc_list

        # word_map_couter = {pair: count for pair, count in pair_map_counter.items() if len(pair[0]) == len(pair[1]) == WORD_LENGHT}
        # pair_map_distribution = np.asarray(list(word_map_couter.values()))
        # pair_map_distribution = pair_map_distribution/np.sum(pair_map_distribution)

        output_dist_base_path = os.path.join(OUTPUT_PATH, f"dist-avg-distance_threshold_{str(dist_threshold).replace('.', 'p')}")
        output_central2context_path = os.path.join(output_dist_base_path, f"central2context.yaml")
        output_pair_map_counter_path = os.path.join(output_dist_base_path, f"pair_map_counter.yaml")

        os.makedirs(output_dist_base_path, exist_ok=True)

        # simplify data for saving
        saving_central2context = {central: list(context) for central, context in central2context.items()}
        saving_pair_map_counter = {'counter': [{'pair': list(pair), 'count': count} for pair, count in pair_map_counter.items()]}

        # dumping pkl is an option but it not convenient to know what's going on in this file
        with open(output_central2context_path, 'w') as fileHandler:
            yaml.safe_dump(saving_central2context, fileHandler, indent=4)
        with open(output_pair_map_counter_path, 'w') as fileHandler:
            yaml.safe_dump(saving_pair_map_counter, fileHandler, indent=4)
