from core.utils.processingUtil import HLAProcessor, PeptideGenerator
from core.exception import InvalidConfigurationException
from core.options.InferenceOption import inferenceParser, MHCToolOption
from core.datasets import DATASET_TO_CLASS_FACTORY, build_inference_pipeline, build_pipeline, MSI011320
from core.models import build_model
from core.optimizer import build_optimizer
from core.options import TrainOption, trainParser
from core.utils import plotTraing
from multiprocessing import shared_memory

import os
import numpy as np
import logging
import tensorflow as tf


def setup_logging(save_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(save_path),
            logging.StreamHandler()
        ]
    )


def main():
    # Tensorflow memory stuffs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # check and validate argument
    args: MHCToolOption = inferenceParser.parse_args()
    MHCToolOption.parseArgs(args)
    isOptionValid, errorMessage = MHCToolOption.validateOption()

    if isOptionValid == False:
        raise InvalidConfigurationException(errorMessage)

    print(f"start running with the config\n{MHCToolOption()}")

    # build preprocessor
    HLAProcessor.load_from_disk(TrainOption.VOCAB_ALLELE_PATH, args.ALLELE_MAPPER_PATH)  # TODO move TrainOption
    PeptideGenerator.load_from_disk(TrainOption.VOCAB_PATH, TrainOption.peptide_lenght_dist_path, TrainOption.NEGATIVE_AMINO_PATH)  # TODO: remove TrainOption

    # build dataset
    batch_size_test = 256  # TODO: move batch size to somewhere else?
    # TODO: parse known allele??
    dataset_size = shared_memory.ShareableList(name=f'CustomCSVDataset_inference_0_dataset_size', sequence=[500000]*3)
    dataset_cls, inference_pipeline = build_inference_pipeline(args, batch_size_test)
    _dataset_size_train, _dataset_size_eval, dataset_size_inference = dataset_size

    # build model
    model = build_model(args=args)
    # TODO: mode MODEL_WEIGHT_PATH to args
    MODEL_WEIGHT_PATHS_TEMP = [
        # For publicly available data
        # 'resources/trained_weight/final_model/final_model_1-5/MHCSeqNet2_1_5.h5',
        # 'resources/trained_weight/final_model/final_model_2-5/MHCSeqNet2_2_5.h5',
        # 'resources/trained_weight/final_model/final_model_3-5/MHCSeqNet2_3_5.h5',
        # 'resources/trained_weight/final_model/final_model_4-5/MHCSeqNet2_4_5.h5',
        # 'resources/trained_weight/final_model/final_model_5-5/MHCSeqNet2_5_5.h5',
        # For publicly available data + SMSNet data
        'resources/trained_weight/final_model_with_smsnetdata/final_model_with_smsnetdata_1-5/MHCSeqNet2_1_5.h5',
        'resources/trained_weight/final_model_with_smsnetdata/final_model_with_smsnetdata_2-5/MHCSeqNet2_2_5.h5',
        'resources/trained_weight/final_model_with_smsnetdata/final_model_with_smsnetdata_3-5/MHCSeqNet2_3_5.h5',
        'resources/trained_weight/final_model_with_smsnetdata/final_model_with_smsnetdata_4-5/MHCSeqNet2_4_5.h5',
        'resources/trained_weight/final_model_with_smsnetdata/final_model_with_smsnetdata_5-5/MHCSeqNet2_5_5.h5',
    ]
    # TODO: properly treat USE_ENSEMBLE flag, need to check if we need to rebuild inference_pipeline
    prediction_list = []
    if args.USE_ENSEMBLE:
        for MODEL_WEIGHT_PATH in MODEL_WEIGHT_PATHS_TEMP:
            model.load_weights(MODEL_WEIGHT_PATH)
            prediction_list.append(model.predict(inference_pipeline, steps=int(dataset_size_inference / batch_size_test), verbose=1))
        prediction = np.mean(prediction_list, axis=0)
    else:
        MODEL_WEIGHT_PATH = MODEL_WEIGHT_PATHS_TEMP[args.MODEL_KF]
        model.load_weights(MODEL_WEIGHT_PATH)
        prediction = model.predict(inference_pipeline, steps=int(dataset_size_inference / batch_size_test), verbose=1)
    base_output_path, _output_fname = os.path.split(args.OUTPUT_DIRECTORY)
    os.makedirs(base_output_path, exist_ok=True)
    dataset_cls.save_result(args.OUTPUT_DIRECTORY, prediction)

    # making prediction of a model
    # subprocess.check_output(['predict.py', 'weight_model_1.h5'])
    # breakpoint()

    # TODO: continue here, setup logger, check if GUI to suppress log
    dataset_size.shm.close()
    dataset_size.shm.unlink()


if __name__ == '__main__':
    main()
    # TODO: remove unnecessary output
    print("finished")
    # breakpoint()
