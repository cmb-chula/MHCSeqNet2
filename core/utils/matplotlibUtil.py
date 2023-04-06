from tensorflow.keras.callbacks import History

import matplotlib.pyplot as plt
import os


def plotTraing(callback_hist: History, k_fold_num=None, save_path=None):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(20, 5*3))
    if 'loss' in callback_hist.history:
        axs[0].plot(callback_hist.epoch, callback_hist.history['loss'], label='loss')
    if 'val_loss' in callback_hist.history:
        axs[0].plot(callback_hist.epoch, callback_hist.history['val_loss'], label='val_loss')
    # plt.xticks(range(len(pair_distribution)), list(pair_map.keys()), rotation=90, fontsize=11)
    axs[0].set_xlim(callback_hist.epoch[0], callback_hist.epoch[-1])
    axs[0].set_title("loss per epoch")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")
    # if save_path == None:
    #     plt.show()
    # else:
    #     fig.savefig(os.path.join(save_path, f"monitor_loss_of_fold_{k_fold_num}.png"), dpi=fig.dpi)

    # fig = plt.figure(figsize=(20, 5))
    if 'acc' in callback_hist.history:
        axs[1].plot(callback_hist.epoch, callback_hist.history['acc'], label='acc')
    if 'val_acc' in callback_hist.history:
        axs[1].plot(callback_hist.epoch, callback_hist.history['val_acc'], label='val_acc')
    # plt.xticks(range(len(pair_distribution)), list(pair_map.keys()), rotation=90, fontsize=11)
    axs[1].set_xlim(callback_hist.epoch[0], callback_hist.epoch[-1])
    axs[1].set_title("acc per epoch")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("acc")
    # if save_path == None:
    #     plt.show()
    # else:
    #     fig.savefig(os.path.join(save_path, f"monitor_acc_of_fold_{k_fold_num}.png"), dpi=fig.dpi)

    # fig = plt.figure(figsize=(20, 5))
    if 'lr' in callback_hist.history:
        axs[2].plot(callback_hist.epoch, callback_hist.history['lr'], label='lr')
    # plt.xticks(range(len(pair_distribution)), list(pair_map.keys()), rotation=90, fontsize=11)
    axs[2].set_xlim(callback_hist.epoch[0], callback_hist.epoch[-1])
    axs[2].set_title("training lr")
    axs[2].set_xlabel("epoch")
    axs[2].set_ylabel("lr")
    plt.tight_layout()
    if save_path == None:
        plt.show()
    else:
        fig.savefig(os.path.join(save_path, f"monitor_of_fold_{k_fold_num}.png"), dpi=fig.dpi)
