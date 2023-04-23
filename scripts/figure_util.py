from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, BboxConnectorPatch

import typing
import pandas as pd

# This is exported from make_figure_auc_few_alleles.py at the bottom (THIS IS OUTDATED THO)
auc_this_work_vs_mhc_flurry_retrain = {'HLA-A*01:01': 0.009982611006239872,
                                       'HLA-A*02:01': 0.034657364779890454,
                                       'HLA-A*02:02': 0.030575288887428997,
                                       'HLA-A*02:03': 0.015069158037806507,
                                       'HLA-A*02:04': 0.024943523764386133,
                                       'HLA-A*02:05': 0.022602306016255458,
                                       'HLA-A*02:06': 0.027803929114801673,
                                       'HLA-A*02:07': 0.013960818561355115,
                                       'HLA-A*02:11': 0.02573714096513169,
                                       'HLA-A*03:01': 0.017278375161111748,
                                       'HLA-A*03:02': 0.09159663865546219,
                                       'HLA-A*11:01': 0.023173202904556045,
                                       'HLA-A*11:02': 0.007955854339737889,
                                       'HLA-A*23:01': 0.03775260890392107,
                                       'HLA-A*24:02': 0.014787963240148083,
                                       'HLA-A*24:06': 0.07568066745676472,
                                       'HLA-A*24:07': 0.008479257994923217,
                                       'HLA-A*24:13': 0.04610583251873579,
                                       'HLA-A*25:01': 0.008351308329230034,
                                       'HLA-A*26:01': 0.0182004546146064,
                                       'HLA-A*29:02': -0.0027924188284746965,
                                       'HLA-A*30:01': 0.02399968198222302,
                                       'HLA-A*30:02': 0.05058206147118738,
                                       'HLA-A*31:01': 0.040936111513117046,
                                       'HLA-A*32:01': 0.04116930915737327,
                                       'HLA-A*33:01': 0.006046606720088832,
                                       'HLA-A*33:03': 0.011597903311511004,
                                       'HLA-A*34:01': 0.013125460807077438,
                                       'HLA-A*34:02': 0.023783125706174357,
                                       'HLA-A*36:01': 0.05783489819654941,
                                       'HLA-A*66:01': 0.034567750579746925,
                                       'HLA-A*66:02': 0.019868651272825777,
                                       'HLA-A*68:01': 0.01143559718728926,
                                       'HLA-A*68:02': 0.08868435266421926,
                                       'HLA-A*69:01': -0.0078924120735967,
                                       'HLA-A*74:01': 0.016546487206629212,
                                       'HLA-B*07:02': -0.003162538770771217,
                                       'HLA-B*07:04': 0.009516034340475765,
                                       'HLA-B*08:01': 0.033776381198727945,
                                       'HLA-B*13:01': 0.0033936702624484116,
                                       'HLA-B*13:02': 0.008160317506903492,
                                       'HLA-B*14:02': 0.020350849182063535,
                                       'HLA-B*14:03': 0.012063492063491887,
                                       'HLA-B*15:01': 0.013890094336724168,
                                       'HLA-B*15:02': 0.03122928920564272,
                                       'HLA-B*15:03': 0.03232743730764087,
                                       'HLA-B*15:10': 0.006676176204259643,
                                       'HLA-B*15:11': 0.02092513141620278,
                                       'HLA-B*15:17': 0.026040509169457327,
                                       'HLA-B*15:18': 0.04999999999999993,
                                       'HLA-B*15:42': 0.016666666666666607,
                                       'HLA-B*18:01': -0.0003580726756819974,
                                       'HLA-B*18:03': -0.0009714549693564667,
                                       'HLA-B*27:01': 0.0010679368758506058,
                                       'HLA-B*27:02': -6.934130583724496e-06,
                                       'HLA-B*27:03': 0.0027532282527593654,
                                       'HLA-B*27:04': 0.0012690733538152088,
                                       'HLA-B*27:05': 0.10972224996696733,
                                       'HLA-B*27:06': 0.004241554318138552,
                                       'HLA-B*27:07': 0.00035338048002053757,
                                       'HLA-B*27:08': -0.005847050442084889,
                                       'HLA-B*27:09': 0.03497757146876834,
                                       'HLA-B*27:10': 0.0,
                                       'HLA-B*35:01': 0.017985154320394825,
                                       'HLA-B*35:02': 0.021164021164021163,
                                       'HLA-B*35:03': 0.007463772005422364,
                                       'HLA-B*35:04': 0.21666666666666656,
                                       'HLA-B*35:06': 0.08333333333333337,
                                       'HLA-B*35:07': 0.008514460851910655,
                                       'HLA-B*35:08': 0.03472532489388125,
                                       'HLA-B*37:01': 0.022509229720205415,
                                       'HLA-B*38:01': 0.029859291885958372,
                                       'HLA-B*38:02': 0.012252258462034149,
                                       'HLA-B*39:01': -0.00405096390596249,
                                       'HLA-B*39:05': 0.0,
                                       'HLA-B*39:06': -0.0025566106647186837,
                                       'HLA-B*39:09': 0.018181818181818077,
                                       'HLA-B*39:24': 0.007156401338363483,
                                       'HLA-B*40:01': 0.009295316256202502,
                                       'HLA-B*40:02': 0.012665673801223964,
                                       'HLA-B*40:06': 0.010529930891885031,
                                       'HLA-B*41:01': -0.0033383705573030165,
                                       'HLA-B*41:02': 0.014583333333333282,
                                       'HLA-B*41:03': 0.01945181581404487,
                                       'HLA-B*41:04': 0.08479020979020979,
                                       'HLA-B*41:05': 0.21030778164924513,
                                       'HLA-B*41:06': 0.1324675324675324,
                                       'HLA-B*42:01': 0.01667175910880525,
                                       'HLA-B*44:02': 0.007119928161059663,
                                       'HLA-B*44:03': 0.001636150341212117,
                                       'HLA-B*44:05': 0.0,
                                       'HLA-B*44:08': 0.017814625850340104,
                                       'HLA-B*44:09': 0.04693833530215641,
                                       'HLA-B*44:27': 0.0031838303628093367,
                                       'HLA-B*44:28': -0.008724800653262532,
                                       'HLA-B*45:01': -0.0017941731092493418,
                                       'HLA-B*46:01': 0.04800141170102201,
                                       'HLA-B*47:01': -0.006327160493827089,
                                       'HLA-B*49:01': 0.005188827402174634,
                                       'HLA-B*50:01': 0.009314148564202207,
                                       'HLA-B*51:01': 0.03435594006792009,
                                       'HLA-B*51:08': -0.010175008614425773,
                                       'HLA-B*52:01': 0.015078495685519222,
                                       'HLA-B*53:01': 0.018660831515545206,
                                       'HLA-B*54:01': 0.025583492452487366,
                                       'HLA-B*55:01': 0.00766499170022672,
                                       'HLA-B*55:02': 0.010611986415513308,
                                       'HLA-B*56:01': 0.016810198662455256,
                                       'HLA-B*57:01': 0.030953726165608986,
                                       'HLA-B*57:03': 0.03698311254984643,
                                       'HLA-B*58:01': 0.01605252482513697,
                                       'HLA-B*58:02': 0.05165100814585166,
                                       'HLA-B*73:01': 0.0006591100793998939,
                                       'HLA-C*01:02': 0.018735896111890527,
                                       'HLA-C*02:02': 0.011703189200429898,
                                       'HLA-C*03:03': 0.044886490306646376,
                                       'HLA-C*03:04': 0.006995253156112935,
                                       'HLA-C*04:01': 0.011873447247937485,
                                       'HLA-C*04:03': 0.02842016385515489,
                                       'HLA-C*05:01': 0.018509296085906013,
                                       'HLA-C*06:02': 0.08274955073955181,
                                       'HLA-C*07:01': 0.04444574131941814,
                                       'HLA-C*07:02': 0.03919760748774381,
                                       'HLA-C*07:04': 0.0482713910446394,
                                       'HLA-C*08:01': 0.016465922849815517,
                                       'HLA-C*08:02': 0.012627925632810011,
                                       'HLA-C*12:02': 0.02603501087921878,
                                       'HLA-C*12:03': 0.008764863093267072,
                                       'HLA-C*14:02': -0.0032358648759067687,
                                       'HLA-C*14:03': 0.005132302280880907,
                                       'HLA-C*15:02': 0.014519615718698975,
                                       'HLA-C*15:05': 0.010869565217391242,
                                       'HLA-C*16:01': 0.026887797663110224,
                                       'HLA-C*17:01': 0.032904372552120975}


def connect_bbox(bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
        }

    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           # loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_patches)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect01(ax1, ax2, xmin, xmax, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
    be marked.

    Parameters
    ----------
    ax1
        The main axes.
    ax2
        The zoomed axes.
    xmin, xmax
        The limits of the colored area in both plot axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)
    trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)

    bbox = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, trans1)
    mybbox2 = TransformedBbox(bbox, trans2)

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

#     ax1.add_patch(bbox_patch1)
#     ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect02(ax1, ax2, **kwargs):
    """
    ax1 : the main axes
    ax1 : the zoomed axes

    Similar to zoom_effect01.  The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

#     ax1.add_patch(bbox_patch1)
#     ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


# def CHART_COLOR_CYCLE(i: int): return ['b', 'g', 'r', 'c', 'm', 'y', ][i % 6]

def CHART_COLOR_CYCLE(i: int): return ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"][i % 9]


def get_traing_data_count_lookup(
    # TRAINING_DATA_CSV_PATH: str = 'resources/datasets/MSI011320_ANTI051821Z_COMBINE/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-###_train.csv',
    TRAINING_DATA_CSV_PATH: str = 'resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-###_train.csv',
    KFOLD: typing.List[int] = [1],
):
    if TRAINING_DATA_CSV_PATH is not None:
        training_count_seires_list: typing.List[pd.Series] = []
        for k in KFOLD:
            fold_training_data: pd.DataFrame = pd.read_csv(TRAINING_DATA_CSV_PATH.replace('###', str(k), 1), index_col=0)
            training_count_seires_list.append(fold_training_data.groupby(by='Allele').size())
        training_data_count_lookup: typing.Dict[str, int] = pd.concat(training_count_seires_list).groupby(by='Allele').sum().to_dict()
    else:
        # for simplicity, if it's unable to load we'll use result form first kfold
        training_data_count_lookup = {'HLA-A*01:01': 9747, 'HLA-A*01:03': 5, 'HLA-A*02:01': 17845, 'HLA-A*02:02': 4084, 'HLA-A*02:03': 2232, 'HLA-A*02:04': 3571, 'HLA-A*02:05': 5487, 'HLA-A*02:06': 2694, 'HLA-A*02:07': 4815, 'HLA-A*02:11': 2727, 'HLA-A*02:14': 19, 'HLA-A*02:20': 21, 'HLA-A*03:01': 6255, 'HLA-A*03:02': 21, 'HLA-A*11:01': 10488, 'HLA-A*11:02': 3419, 'HLA-A*23:01': 4464, 'HLA-A*24:02': 7863, 'HLA-A*24:06': 428, 'HLA-A*24:07': 1692, 'HLA-A*24:13': 276, 'HLA-A*25:01': 1261, 'HLA-A*26:01': 2077, 'HLA-A*26:03': 1, 'HLA-A*29:02': 8124, 'HLA-A*29:06': 1, 'HLA-A*30:01': 2248, 'HLA-A*30:02': 3346, 'HLA-A*30:03': 6, 'HLA-A*30:04': 4, 'HLA-A*30:14': 29, 'HLA-A*31:01': 2451, 'HLA-A*32:01': 4829, 'HLA-A*33:01': 3031, 'HLA-A*33:03': 3647, 'HLA-A*34:01': 2733, 'HLA-A*34:02': 4535, 'HLA-A*36:01': 3415, 'HLA-A*66:01': 2612, 'HLA-A*66:02': 32, 'HLA-A*68:01': 4689, 'HLA-A*68:02': 4602, 'HLA-A*69:01': 767, 'HLA-A*74:01': 3251, 'HLA-B*07:02': 17735, 'HLA-B*07:04': 1680, 'HLA-B*08:01': 6647, 'HLA-B*13:01': 4665, 'HLA-B*13:02': 3793, 'HLA-B*14:01': 3, 'HLA-B*14:02': 5223, 'HLA-B*14:03': 21, 'HLA-B*15:01': 15956, 'HLA-B*15:02': 3781, 'HLA-B*15:03': 3432, 'HLA-B*15:08': 28, 'HLA-B*15:09': 14, 'HLA-B*15:10': 1835, 'HLA-B*15:11': 130, 'HLA-B*15:13': 11, 'HLA-B*15:16': 13, 'HLA-B*15:17': 2140, 'HLA-B*15:18': 8, 'HLA-B*15:42': 3, 'HLA-B*18:01': 3226, 'HLA-B*18:03': 238, 'HLA-B*27:01': 5288, 'HLA-B*27:02': 3444, 'HLA-B*27:03': 1153, 'HLA-B*27:04': 1370, 'HLA-B*27:05': 60992, 'HLA-B*27:06': 1293, 'HLA-B*27:07': 2270, 'HLA-B*27:08': 2248, 'HLA-B*27:09': 6417, 'HLA-B*27:10': 12, 'HLA-B*35:01': 7475, 'HLA-B*35:02': 45, 'HLA-B*35:03': 3291, 'HLA-B*35:04': 15,
                                      'HLA-B*35:06': 11, 'HLA-B*35:07': 2431, 'HLA-B*35:08': 1151, 'HLA-B*37:01': 6316, 'HLA-B*38:01': 3212, 'HLA-B*38:02': 3677, 'HLA-B*39:01': 1922, 'HLA-B*39:05': 9, 'HLA-B*39:06': 92, 'HLA-B*39:09': 10, 'HLA-B*39:10': 8, 'HLA-B*39:24': 278, 'HLA-B*40:01': 4588, 'HLA-B*40:02': 17309, 'HLA-B*40:06': 2847, 'HLA-B*41:01': 476, 'HLA-B*41:02': 21, 'HLA-B*41:03': 93, 'HLA-B*41:04': 52, 'HLA-B*41:05': 42, 'HLA-B*41:06': 19, 'HLA-B*42:01': 4140, 'HLA-B*44:01': 1, 'HLA-B*44:02': 6972, 'HLA-B*44:03': 4871, 'HLA-B*44:05': 4, 'HLA-B*44:08': 50, 'HLA-B*44:09': 226, 'HLA-B*44:27': 199, 'HLA-B*44:28': 134, 'HLA-B*45:01': 2678, 'HLA-B*46:01': 3187, 'HLA-B*47:01': 31, 'HLA-B*49:01': 8097, 'HLA-B*50:01': 1710, 'HLA-B*50:02': 2, 'HLA-B*51:01': 4441, 'HLA-B*51:02': 5, 'HLA-B*51:08': 673, 'HLA-B*52:01': 1928, 'HLA-B*53:01': 2656, 'HLA-B*54:01': 1412, 'HLA-B*55:01': 2019, 'HLA-B*55:02': 1821, 'HLA-B*56:01': 2571, 'HLA-B*57:01': 19940, 'HLA-B*57:02': 8, 'HLA-B*57:03': 7147, 'HLA-B*58:01': 6648, 'HLA-B*58:02': 1367, 'HLA-B*73:01': 223, 'HLA-B*78:01': 4, 'HLA-C*01:02': 2950, 'HLA-C*02:02': 7191, 'HLA-C*03:03': 7147, 'HLA-C*03:04': 5191, 'HLA-C*03:32': 23, 'HLA-C*04:01': 9673, 'HLA-C*04:03': 1432, 'HLA-C*05:01': 5453, 'HLA-C*06:01': 7, 'HLA-C*06:02': 6560, 'HLA-C*07:01': 4574, 'HLA-C*07:02': 4254, 'HLA-C*07:04': 4215, 'HLA-C*08:01': 2352, 'HLA-C*08:02': 9122, 'HLA-C*12:02': 11942, 'HLA-C*12:03': 3388, 'HLA-C*12:04': 5, 'HLA-C*14:02': 2561, 'HLA-C*14:03': 3538, 'HLA-C*15:01': 1, 'HLA-C*15:02': 2015, 'HLA-C*15:05': 8, 'HLA-C*16:01': 4247, 'HLA-C*17:01': 1672, 'HLA-E*01:01': 16, 'HLA-E*01:03': 662, 'HLA-G*01:01': 2973, 'HLA-G*01:03': 970, 'HLA-G*01:04': 1068}
    return training_data_count_lookup


def get_few_allele_set(
    # TRAINING_DATA_CSV_PATH: str = 'resources/datasets/MSI011320_ANTI051821Z_COMBINE/HLA_classI_MS_dataset_011320_antigen_information_051821_rev1_processed_kf-###_train.csv',
    TRAINING_DATA_CSV_PATH: str = 'resources/datasets/MSI011320/HLA_classI_MS_dataset_011320_processed_kf-###_train.csv',
    KFOLD: typing.List[int] = [1, 2, 3, 4, 5],
    FEW_ALLELE_THRESHOLD: typing.Tuple[int, int] = [0, 200],  # exclusive
):
    training_data_count_lookup = get_traing_data_count_lookup(TRAINING_DATA_CSV_PATH, KFOLD)
    return {allele for allele, count in training_data_count_lookup.items() if FEW_ALLELE_THRESHOLD[0] < count < FEW_ALLELE_THRESHOLD[1]}
