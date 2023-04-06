import typing
from PyQt6.QtWidgets import QMessageBox
# from PyQt6.QtWidgets import QDialog, QFileDialog
# from functools import reduce


# def openFileDialog(mouseEv, caption: str, *options: QFileDialog.Option, **kwargs):
#     dialog = QFileDialog()
#     dialog.setDefaultSuffix('csv')
#     dialog.setNameFilters(['CSV (*.csv)'])
#     dialog.setAcceptMode(QFileDialog.AcceptOpen)
#     options = None if len(options) == 0 else reduce(lambda x, y: x | y, options)
#     # options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly
#     QFileDialog.getOpenFileName(dialog, options=options, **kwargs)
#     if dialog.exec_() == QDialog.Accepted:
#         return dialog.selectedFiles()
#     else:
#         return None


def showError(title: str = "Error", icon: QMessageBox.Icon = QMessageBox.Icon.Critical, sumText: str = "An error occour", infoText: typing.Optional[str] = None, detailedText: typing.Optional[str] = None, standDardButtons: typing.Union['QMessageBox.StandardButtons', 'QMessageBox.StandardButton'] = QMessageBox.StandardButton.Ok):
    msg = QMessageBox()
    msg.setWindowTitle(title)
    msg.setIcon(icon)
    msg.setText(sumText)
    if infoText is not None:
        msg.setInformativeText(infoText)
    if detailedText is not None:
        msg.setDetailedText(detailedText)
    msg.setStandardButtons(standDardButtons)
    return msg.exec_()
