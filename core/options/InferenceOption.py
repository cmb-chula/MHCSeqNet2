from __future__ import annotations

import os
from translation import Lang
from typing import List, Literal
import typing
import argparse


class MHCToolOption(object):
    MODE: Literal['CSV', 'CROSS'] = 'CSV'
    CSV_PATH: str = None
    PEPTIDE_COLUMN_NAME: str = None
    ALLELE_COLUMN_NAME: str = None
    PEPTIDE_PATH: str = None
    ALLELE_PATH: str = None
    # RUN config
    IGNORE_UNKNOW: bool = True
    LOG_UNKNOW: bool = False
    LOG_UNKNOW_PATH: str = "log.txt"
    MODEL_KF: int = 0
    GPU_ID: int = -1
    USE_ENSEMBLE: bool = False
    MODEL_TYPE: Literal['MHCSeqNet2', 'MHCSeqNet2_GRUPeptide', 'GloVeFastText', 'MultiHeadGloVeFastTextSplit', 'MultiHeadGloVeFastTextJointed'] = 'MHCSeqNet2'
    ALLELE_MAPPER_PATH: str = "resources/allele_mapper"
    OUTPUT_DIRECTORY: str = "output.csv"
    TEMP_FILE_PATH: str = "temp.csv"
    SUPPRESS_LOG: bool = True  # this is the only one that is normally setted to False in mhctool.py

    def __repr__(self) -> str:
        return (
            "MHCToolOption:\n"
            f"  MODE: {MHCToolOption.MODE}\n"
            f"  CSV_PATH: {MHCToolOption.CSV_PATH}\n"
            f"  PEPTIDE_COLUMN_NAME: {MHCToolOption.PEPTIDE_COLUMN_NAME}\n"
            f"  ALLELE_COLUMN_NAME: {MHCToolOption.ALLELE_COLUMN_NAME}\n"
            f"  PEPTIDE_PATH: {MHCToolOption.PEPTIDE_PATH}\n"
            f"  ALLELE_PATH: {MHCToolOption.ALLELE_PATH}\n"
            f"  IGNORE_UNKNOW: {MHCToolOption.IGNORE_UNKNOW}\n"
            f"  LOG_UNKNOW: {MHCToolOption.LOG_UNKNOW}\n"
            f"  LOG_UNKNOW_PATH: {MHCToolOption.LOG_UNKNOW_PATH}\n"
            f"  MODEL_KF: {MHCToolOption.MODEL_KF}\n"
            f"  GPU_ID: {MHCToolOption.GPU_ID}\n"
            f"  USE_ENSEMBLE: {MHCToolOption.USE_ENSEMBLE}\n"
            f"  MODEL_TYPE: {MHCToolOption.MODEL_TYPE}\n"
            f"  ALLELE_MAPPER_PATH: {MHCToolOption.ALLELE_MAPPER_PATH}\n"
            f"  OUTPUT_DIRECTORY: {MHCToolOption.OUTPUT_DIRECTORY}\n"
            f"  TEMP_FILE_PATH: {MHCToolOption.TEMP_FILE_PATH}\n"
            f"  SUPPRESS_LOG: {MHCToolOption.SUPPRESS_LOG}\n"
        ).strip()

    @classmethod
    def validateOption(cls) -> typing.Tuple[bool, typing.Optional[str]]:
        if cls.MODE == 'CSV':
            if cls.CSV_PATH is None or os.path.exists(cls.CSV_PATH) is False:
                return False, f"CSV_PATH is invalid or the file does not exist"
        elif cls.MODE == 'CROSS':
            if cls.PEPTIDE_PATH is None or os.path.exists(cls.PEPTIDE_PATH) is False:
                return False, f"PEPTIDE_PATH is invalid or the file does not exist"
            elif cls.ALLELE_PATH is None or os.path.exists(cls.ALLELE_PATH) is False:
                return False, f"ALLELE_PATH is invalid or the file does not exist"
        else:
            return False, f"Unknown mode {cls.MODE}"

        return True, None

    @classmethod
    def parseArgs(cls, args: MHCToolOption) -> None:
        """
        args is from argparser.parse_args(),
        use to fill the class variable
        """
        for prop in cls.__annotations__.keys():
            setattr(cls, prop, getattr(args, prop))

    @classmethod
    def getArgumentsString(cls, sep: typing.Literal[' ', '='] = '=') -> List[str]:
        return [f"--{propName}{sep}{propValue}" if propType != "bool" else f"--{propName}" for propName, propType in cls.__annotations__.items() if (propValue := getattr(cls, propName, None)) is not None]


inferenceParser = argparse.ArgumentParser(description='MHCTool')

# TODO: should I add dry-run
# Normal args
inferenceParser.add_argument('--MODE', default=MHCToolOption.MODE, choices=['CSV', 'CROSS'], help=f'Mode `CSV` or `CROSS`\n{Lang.MODE}')
inferenceParser.add_argument('--CSV_PATH', default=MHCToolOption.CSV_PATH, type=str, help="path directory to input csv when use `--MODE CSV`")
inferenceParser.add_argument('--PEPTIDE_COLUMN_NAME', default=MHCToolOption.PEPTIDE_COLUMN_NAME, type=str, help='the column name which containing the peptide')
inferenceParser.add_argument('--ALLELE_COLUMN_NAME', default=MHCToolOption.ALLELE_COLUMN_NAME, type=str, help='the column name which containing the allele')
inferenceParser.add_argument('--PEPTIDE_PATH', default=MHCToolOption.PEPTIDE_PATH, type=str, help='path directory to input peptide when use `--MODE CROSS`')
inferenceParser.add_argument('--ALLELE_PATH', default=MHCToolOption.ALLELE_PATH, type=str, help='path directory to input allele when use `--MODE CROSS`')
inferenceParser.add_argument('--IGNORE_UNKNOW', default=MHCToolOption.IGNORE_UNKNOW, action='store_true', help='if setted it will skip the unknown')
inferenceParser.add_argument('--LOG_UNKNOW', default=MHCToolOption.LOG_UNKNOW, action='store_true', help='if setted it will log the unknown that was skipped')
inferenceParser.add_argument('--LOG_UNKNOW_PATH', default=MHCToolOption.LOG_UNKNOW_PATH, type=str, help='the file which the unknow will be logged to')
inferenceParser.add_argument('--MODEL_KF', default=MHCToolOption.MODEL_KF, type=int, help='specify model weight to use, if not using ensemble')
inferenceParser.add_argument('--GPU_ID', default=MHCToolOption.GPU_ID, type=int, help='default GPU, you can specify a GPU to be used by given a number i.e, `--GPU_ID 0`')
inferenceParser.add_argument('--USE_ENSEMBLE', default=MHCToolOption.USE_ENSEMBLE, action='store_true',
                             help='Run the result multiple times on multiple models and use the average as the score')
inferenceParser.add_argument('--MODEL_TYPE', default=MHCToolOption.MODEL_TYPE, type=str,
                             choices=['MHCSeqNet2', 'MHCSeqNet2_GRUPeptide', 'GloVeFastText', 'MultiHeadGloVeFastTextSplit', 'MultiHeadGloVeFastTextJointed'], help='specify model to use')
inferenceParser.add_argument('--ALLELE_MAPPER_PATH', default=MHCToolOption.ALLELE_MAPPER_PATH, type=str,
                             help='path to the folder that contain yaml file needed for the tool.\nYou can use this to add a new allele, please visit readme for more')
inferenceParser.add_argument('--OUTPUT_DIRECTORY', default=MHCToolOption.OUTPUT_DIRECTORY, type=str, help='where to save the final result to (only .csv or .tsv). **The output column name is `Prediction` where 0 mean bind and 1 mean otherwise**')
inferenceParser.add_argument('--TEMP_FILE_PATH', default=MHCToolOption.TEMP_FILE_PATH, type=str,
                             help='path to intermediate result file\nto maintain system compatibility and stability, the program need to store intermediate result.')
# Internal args
inferenceParser.add_argument('--SUPPRESS_LOG', default=False, action='store_true', help='use to suppress log, useful only for running from gui')
