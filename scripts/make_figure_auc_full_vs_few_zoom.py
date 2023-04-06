from __future__ import annotations

import sys  # nopep8
sys.path.append('.')  # nopep8

from scripts.figure_util import get_few_allele_set, get_traing_data_count_lookup, zoom_effect01, CHART_COLOR_CYCLE

import os
import typing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from sklearn.metrics import roc_curve, auc
from matplotlib.axes import Axes

FIG_PATH = 'figures/fig1_auc_overall_vs_few'
FIGSIZE = (12, 12)
TOTAL_KFOLD = 5
# 'class', 'name', 'pred column', 'label column', 'Need flip label' ,'Need flip prediction', 'path to all fold folder'
KFOLD_RESULT_PATH: typing.List[typing.Tuple[str, str, str, str, bool, bool, str]] = [
    ('ExperimentalResult', 'this work', 'Prediction', 'isGenerated', True, True, 'resources/trained_weight/final_model'),
]
ALLELE_GROUP_COLUMN = 'ALLELE_GROUP'
ALLELE_GROUP_NAME = 'FEW_ALLELE_GROUPED'


def get_chart_color(index: int, name: str):
    lookup = {
        'combine old&new Fix len dist': "#e60049",
        'combine old&new 3D': "#0bb4ff",
        'this work': "#1e45c7",
        'both pretrain FIX len dist long': "#e6d800",
        'no pretrain': "#1e45c7",
        'mhcflurry re-train': "#dc0ab4",
        'mhcflurry pretrain': "#dc0ab4",
        'mhcseqnet re-train': "#b3d4ff",
        'netmhcpan pretrain': "#00bfa0",
    }
    return lookup.get(name, CHART_COLOR_CYCLE(index))


def get_chart_line_style(name: str):
    lookup = {
        'combine old&new Fix len dist': "solid",
        'combine old&new 3D': "solid",
        'this work': "solid",
        'both pretrain FIX len dist long': "solid",
        'no pretrain': "dashed",
        'mhcflurry re-train': "solid",
        'mhcflurry pretrain': "dashed",
        'mhcseqnet re-train': "solid",
        'netmhcpan pretrain': "solid",
    }
    return lookup.get(name, "solid")


@dataclass
class ROCResult:
    tpr: typing.List[float]
    fpr: typing.List[float]
    thresholds: typing.List[float]
    auc: float = field(default=None, init=False)
    tpr_at_0p5fpr: float = field(default=None, init=False)

    def __post_init__(self):
        self.auc = auc(self.fpr, self.tpr)
        self.tpr_at_0p5fpr = self.tpr[np.argmin(np.abs(self.fpr-0.05))]


@dataclass
class RunResult:
    name: str
    pred_column: str
    label_column: str
    need_flipping_label: bool
    need_flipping_pred: bool
    path: str
    chart_color: str
    data_frame: pd.DataFrame = field(default=None, init=False, repr=False)
    result_overall: ROCResult = field(default=None, init=False)
    result_few_allele: ROCResult = field(default=None, init=False)

    def cal_auc(self):
        # calculate auc, noted that  we ignore nan or missing value
        overall_allele_mask = np.logical_and.reduce([self.data_frame[ALLELE_GROUP_COLUMN] != ALLELE_GROUP_NAME, np.logical_not(self.data_frame[self.pred_column].isnull())])
        fpr, tpr, thresholds = roc_curve(y_true=self.data_frame.loc[overall_allele_mask, self.label_column],
                                         y_score=self.data_frame.loc[overall_allele_mask, self.pred_column])
        self.result_overall = ROCResult(tpr=tpr, fpr=fpr, thresholds=thresholds)
        few_allele_mask = np.logical_and.reduce([self.data_frame[ALLELE_GROUP_COLUMN] == ALLELE_GROUP_NAME, np.logical_not(self.data_frame[self.pred_column].isnull())])
        fpr, tpr, thresholds = roc_curve(y_true=self.data_frame.loc[few_allele_mask, self.label_column],
                                         y_score=self.data_frame.loc[few_allele_mask, self.pred_column])
        self.result_few_allele = ROCResult(tpr=tpr, fpr=fpr, thresholds=thresholds)


@dataclass
class ExperimentalResult(RunResult):

    def __post_init__(self):
        RUN_NAME = os.path.basename(self.path)
        all_df_list = []
        for kfold in range(1, TOTAL_KFOLD + 1):
            if not self.path.endswith('.csv'):
                pred_path = os.path.join(self.path, f"{RUN_NAME}_{kfold}-{TOTAL_KFOLD}/pred_test.csv")
            else:
                # path is csv, directly load the csv
                pred_path = self.path
            assert os.path.exists(pred_path), f"Unable to locate predition file not found at {pred_path}"
            kfold_df = pd.read_csv(pred_path, index_col=0)
            # post-process dataframe
            kfold_df.loc[:, "kfold"] = kfold
            if self.need_flipping_label:
                kfold_df.loc[:, self.label_column] = 1 - kfold_df[self.label_column]
            if self.need_flipping_pred:
                kfold_df.loc[:, self.pred_column] = 1 - kfold_df[self.pred_column]
            kfold_df.loc[:, ALLELE_GROUP_COLUMN] = kfold_df['Allele'].apply(lambda x: x if x not in few_allele_set else ALLELE_GROUP_NAME)
            all_df_list.append(kfold_df)
        self.data_frame = pd.concat(all_df_list, ignore_index=True)
        self.cal_auc()


@dataclass
class MHCFlurryKFoldResult(RunResult):
    """
    This support MHCFlurry's prediction format

    ```python
    predictor.predict_to_dataframe(...)
    ```

    I used this for the result of re-train MHCFlurry
    """

    def __post_init__(self):
        RUN_NAME = os.path.basename(self.path)
        TOTAL_KFOLD = 4
        all_df_list = []
        for kfold in range(1, TOTAL_KFOLD + 1):
            if not self.path.endswith('.csv'):
                pred_path = os.path.join(self.path, f"{RUN_NAME}_fold_{kfold-1}.csv")
            else:
                # path is csv, directly load the csv
                pred_path = self.path
            assert os.path.exists(pred_path), f"Unable to locate predition file not found at {pred_path}"
            kfold_df = pd.read_csv(pred_path, index_col=False)
            # post-process dataframe
            kfold_df.loc[:, "kfold"] = kfold
            # rename column
            kfold_df.rename(columns={"allele": "Allele", "peptide": "Peptide"}, inplace=True)
            if self.need_flipping_label:
                kfold_df.loc[:, self.label_column] = 1 - kfold_df[self.label_column]
            if self.need_flipping_pred:
                kfold_df.loc[:, self.pred_column] = 1 - kfold_df[self.pred_column]
            kfold_df.loc[:, ALLELE_GROUP_COLUMN] = kfold_df['Allele'].apply(lambda x: x if x not in few_allele_set else ALLELE_GROUP_NAME)
            all_df_list.append(kfold_df)
        self.data_frame = pd.concat(all_df_list, ignore_index=True)
        self.cal_auc()


@dataclass
class NonKFoldResult(RunResult):
    def __post_init__(self):
        RUN_NAME = os.path.basename(self.path)
        if not self.path.endswith('.csv'):
            pred_path = os.path.join(self.path, f'pred_test_{RUN_NAME}.csv')
        else:
            # path is csv, directly load the csv
            pred_path = self.path
        assert os.path.exists(pred_path), f"Unable to locate predition file not found at {pred_path}"
        pred_df = pd.read_csv(pred_path, index_col=0)
        if self.need_flipping_label:
            pred_df.loc[:, self.label_column] = 1 - pred_df[self.label_column]
        if self.need_flipping_pred:
            pred_df.loc[:, self.pred_column] = 1 - pred_df[self.pred_column]
        pred_df.loc[:, ALLELE_GROUP_COLUMN] = pred_df['Allele'].apply(lambda x: x if x not in few_allele_set else ALLELE_GROUP_NAME)
        self.data_frame = pred_df
        self.cal_auc()


run_result_mapper: typing.Dict[str, typing.Union[ExperimentalResult, NonKFoldResult, MHCFlurryKFoldResult]] = {
    'ExperimentalResult': ExperimentalResult,
    'NonKFoldResult': NonKFoldResult,
    'MHCFlurryKFoldResult': MHCFlurryKFoldResult,
}

if __name__ == "__main__":
    TRAINING_DATA_COUNT_LOOKUP = get_traing_data_count_lookup(KFOLD=[1, 2, 3, 4, 5])
    few_allele_set = get_few_allele_set(KFOLD=[1, 2, 3, 4, 5])

    experiment_result: typing.Dict[str, ExperimentalResult] = {
        data_name: run_result_mapper[run_result_type](name=data_name,
                                                      pred_column=pred_column,
                                                      label_column=label_column,
                                                      need_flipping_label=data_need_flip_label,
                                                      need_flipping_pred=data_need_flip_pred,
                                                      path=data_path,
                                                      chart_color=get_chart_color(i, data_name)) for i, (run_result_type, data_name, pred_column, label_column, data_need_flip_label, data_need_flip_pred, data_path) in enumerate(KFOLD_RESULT_PATH)
    }

    fig = plt.figure(figsize=FIGSIZE)
    plt.rcParams.update({'font.size': 12.5}) # Original is 10
    axs: typing.Tuple[Axes, Axes, Axes, Axes] = plt.subplot(221), plt.subplot(223), plt.subplot(222), plt.subplot(224)

    # plot overall graph data
    for i in range(0, 2, 1):
        ax: Axes = axs[i]
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'(C) ROC for overall alleles')
        for exp_name, exp in sorted(experiment_result.items(), key=lambda x: x[1].result_overall.auc):
            exp_result = exp.result_overall
            ax.plot(exp_result.fpr, exp_result.tpr, lw=2,
                    label=f"{exp_name} (AUC={exp_result.auc:0.2f}, TPR={exp_result.tpr_at_0p5fpr:0.2f})", color=exp.chart_color, linestyle=get_chart_line_style(exp.name))
            # ax.plot([0.05, 0.05], [0, 2], color='k', lw=1,
            #         label=f"0.05 FPR, {exp_name} TPR = {exp_result.tpr_at_0p5fpr:0.6f}")
        ax.plot([0.05, 0.05], [0, 2], color='k', lw=1)
        legend = ax.legend(loc="best")
        if i == 0:
            ax.set_title(f'(A) <0.05 FPR region for overall alleles')

    # plot low data
    for i in range(2, 4, 1):
        ax: Axes = axs[i]
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'(D) ROC for allele with few data')
        for exp_name, exp in sorted(experiment_result.items(), key=lambda x: x[1].result_few_allele.auc):
            exp_result = exp.result_few_allele
            ax.plot(exp_result.fpr, exp_result.tpr, lw=2,
                    label=f"{exp_name} (AUC={exp_result.auc:0.2f}, TPR={exp_result.tpr_at_0p5fpr:0.2f})", color=exp.chart_color, linestyle=get_chart_line_style(exp.name))
            # ax.plot([0.05, 0.05], [0, 2], color='k', lw=1,
            #         label=f"0.05 FPR, {exp_name} TPR = {exp_result.tpr_at_0p5fpr:0.6f}")
        ax.plot([0.05, 0.05], [0, 2], color='k', lw=1)
        legend = ax.legend(loc="best")
        if i == 2:
            ax.set_title(f'(B) <0.05 FPR region for allele with few data')

    axs[0].set_xlim(0, 0.05)
    axs[0].set_ylim(0.4, 1.0)
    axs[1].set_xlim(0, 1)
    zoom_effect01(axs[0], axs[1], 0.0, 0.05)

    axs[2].set_xlim(0, 0.05)
    axs[2].set_ylim(0.4, 1.0)
    axs[3].set_xlim(0, 1)
    zoom_effect01(axs[2], axs[3], 0.0, 0.05)

    os.makedirs(os.path.split(FIG_PATH)[0], exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}.png")
    plt.savefig(f"{FIG_PATH}.eps")
