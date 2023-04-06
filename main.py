from translation import Lang
import typing
from core.pyqtUtil import showError
from core.SubProcessWorker import SubProcessWorker
from core.options.InferenceOption import MHCToolOption
import os
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton, QTableWidget, QTabWidget,
                             QTableWidgetItem, QWidget)
import pandas as pd

GPU_PHYSICAL_DEVICES: typing.List['tf.config.PhysicalDevice'] = []


class MHCToolGUI(QDialog):
    def __init__(self, parent=None):
        super(MHCToolGUI, self).__init__(parent)
        global GPU_PHYSICAL_DEVICES
        try:
            # TODO: force use tensorflow
            import tensorflow as tf
            GPU_PHYSICAL_DEVICES = tf.config.list_physical_devices('GPU')
        except ImportError:
            GPU_PHYSICAL_DEVICES = []

        # TODO: change name
        applicationTitleLabel = QLabel("MHCSeqNet2 Launcher")
        self.runProcessPushButton = QPushButton("Process")
        self.runProcessPushButton.setDefault(True)

        self.createInputGroupBox()
        self.createProgressBar()
        self.createRunConfigGroupBox()

        topLayout = QHBoxLayout()
        topLayout.addWidget(applicationTitleLabel)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.inputTabWidget, 1, 0, 1, 2)
        mainLayout.addWidget(self.runConfigGroupBox, 2, 0)
        mainLayout.addWidget(self.runProcessPushButton, 3, 0)
        mainLayout.addWidget(self.progressBar, 4, 0, 1, 2)
        self.setLayout(mainLayout)

        self.runProcessPushButton.clicked.connect(self.processMHC)
        self.setWindowTitle("TODO")  # TODO: change name

    def processMHC(self):
        isOptionValid, errorMessage = MHCToolOption.validateOption()
        if isOptionValid is False:
            showError(infoText=errorMessage, detailedText=f"{MHCToolOption()}")
            return

        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setRange(0, 0)
        self.mhcWorker = SubProcessWorker(command="python mhctool.py")  # no parent! TODO: change to how mhc use
        self.thread = QThread()  # no parent!
        # TODO set icon and neame MHCSeqNETv2

        def processFinishedHandler(isSuccessfull: bool):
            self.thread.quit()
            self.runProcessPushButton.setDisabled(False)
            self.progressBar.setRange(0, 100)
            if isSuccessfull:
                # what to do when there is no error
                self.progressBar.setValue(100)
                self.progressBar.setTextVisible(True)
                print(f"Output:\n{self.mhcWorker.output.decode('utf-8')}\n\n\nError:{self.mhcWorker.error_output.decode('utf-8')}")
            else:
                # error because spawn process error
                if self.mhcWorker.error_output[:5] == b'self:':
                    # print('Error because spawn process failed')
                    errorInfoText = "Unable to start processing process"
                    self.mhcWorker.error_output = self.mhcWorker.error_output[5:]
                else:  # error because MHC-TOOLS throw error
                    # print('Error because program error')
                    errorInfoText = "Processing process is stopped unfinished"
                showError(infoText=errorInfoText, detailedText=f"The details are as follows:\n{self.mhcWorker.error_output.decode('utf-8')}")

        # 3 - Move the Worker object to the Thread object
        self.mhcWorker.moveToThread(self.thread)

        # 4 - Connect Worker Signals to the Thread slots
        self.mhcWorker.finished.connect(processFinishedHandler)

        # 5 - Connect Thread started signal to Worker operational slot method
        self.thread.started.connect(self.mhcWorker.procCounter)

        # * - Thread finished signal will close the app if you want!
        # self.thread.finished.connect(app.exit)

        # 6 - Start the thread
        self.thread.start()
        self.runProcessPushButton.setDisabled(True)

    def createInputGroupBox(self):
        self.inputTabWidget = QTabWidget()
        # self.inputTabWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)

        # CSV mode
        filePathLabel = QLabel()
        filePathLabel.setText(Lang.CSV.DEFAULT_BROWS_TEXT)
        filePathLabel.setToolTip(Lang.CSV.HELP)
        filePathLabel.setWhatsThis(Lang.CSV.HELP)
        filePathEdit = QLineEdit()
        filePathEdit.setToolTip(Lang.CSV.HELP)
        filePathEdit.setWhatsThis(Lang.CSV.HELP)
        filePathEdit.setReadOnly(True)
        pairModeTab = QWidget()
        filePathLayout = QGridLayout()
        filePathLayout.addWidget(filePathLabel, 0, 0, 1, 1)
        filePathLayout.addWidget(filePathEdit, 1, 0, 1, 1)

        previewTableLabel = QLabel()
        previewTableLabel.setText(Lang.CSV.PREVIEW_TITLE)
        previewTableLabel.setToolTip(Lang.CSV.PREVIEW_HELP)
        previewTableLabel.setWhatsThis(Lang.CSV.PREVIEW_HELP)
        filePreviewTableWidget = QTableWidget(10, 2)
        filePreviewTableWidget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        filePreviewTableWidget.setToolTip(Lang.CSV.PREVIEW_HELP)
        filePreviewTableWidget.setWhatsThis(Lang.CSV.PREVIEW_HELP)
        filePreviewTableWidget.setItem(0, 0, QTableWidgetItem("Peptide"))
        filePreviewTableWidget.setItem(0, 1, QTableWidgetItem("Allele"))
        filePreviewTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        filePathLayout.addWidget(previewTableLabel)
        filePathLayout.addWidget(filePreviewTableWidget)
        pairModeTab.setLayout(filePathLayout)

        def validateAndPreviewCSVFile(fileURL: str):
            MHCToolOption.PEPTIDE_COLUMN_NAME = None
            MHCToolOption.ALLELE_COLUMN_NAME = None
            with open(fileURL, 'r') as fileHandler:
                sep = ',' if os.path.splitext(fileURL)[1].lower() == '.csv' else '\t'
                columnNames = set([tok.strip() for tok in fileHandler.readline().split(sep)])
                peptide_column_names = {"peptide", "Peptide", "PEPTIDE"}.intersection(columnNames)
                allele_column_names = {"allele", "Allele", "ALLELE"}.intersection(columnNames)
                if len(peptide_column_names) > 0:
                    MHCToolOption.PEPTIDE_COLUMN_NAME = peptide_column_names.pop()
                if len(allele_column_names) > 0:
                    MHCToolOption.ALLELE_COLUMN_NAME = allele_column_names.pop()
                if (MHCToolOption.PEPTIDE_COLUMN_NAME == None) != (MHCToolOption.ALLELE_COLUMN_NAME == None):
                    # unsupport
                    showError(sumText=Lang.CSV.VALIDATE_ERROR_SUM_TEXT,
                              infoText=Lang.CSV.VALIDATE_ERROR_INFO,
                              detailedText=f"{Lang.CSV.VALIDATE_ERROR_DETAIL1}peptide={MHCToolOption.PEPTIDE_COLUMN_NAME or 'MISSING'}, allele={MHCToolOption.ALLELE_COLUMN_NAME or 'MISSING'}.\n"
                              f"{Lang.CSV.VALIDATE_ERROR_DETAIL2}")
                    return
            # let's preview
            isHasColumnName = None if (MHCToolOption.PEPTIDE_COLUMN_NAME == None) and (MHCToolOption.ALLELE_COLUMN_NAME == None) else 0
            preview_df: pd.DataFrame = pd.read_csv(fileURL, sep, header=isHasColumnName, nrows=10, skipinitialspace=True)
            filePreviewTableWidget.clearContents()
            filePreviewTableWidget.setItem(0, 0, QTableWidgetItem("Peptide"))
            filePreviewTableWidget.setItem(0, 1, QTableWidgetItem("Allele"))
            for ir, (_ir, row) in enumerate(preview_df.iterrows()):
                filePreviewTableWidget.setItem(ir + 1, 0, QTableWidgetItem(row[MHCToolOption.PEPTIDE_COLUMN_NAME or 0]))
                filePreviewTableWidget.setItem(ir + 1, 1, QTableWidgetItem(row[MHCToolOption.ALLELE_COLUMN_NAME or 1]))

            filePathEdit.setText(fileURL)
            MHCToolOption.CSV_PATH = fileURL

        def browsFileCsv(_mouseEvent: QMouseEvent):
            if not filePathEdit.underMouse():
                return
            dialog = QFileDialog()
            dialog.setWindowTitle(Lang.CSV.DEFAULT_BROWS_TEXT)
            dialog.setDefaultSuffix('csv')
            dialog.setNameFilters(['CSV (*.csv)', 'TSV (*.tsv)'])
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                fileURL = dialog.selectedFiles()[0]
                validateAndPreviewCSVFile(fileURL)
        # def setCsvPath(changedText: str):
        #     MHCToolOption.CSV_PATH = changedText
        filePathEdit.mouseReleaseEvent = browsFileCsv
        # filePathEdit.textChanged.connect(setCsvPath)

        # cross mode
        # peptide
        peptideFilePathLabel = QLabel()
        peptideFilePathLabel.setText(Lang.CROSS.PEPTIDE_BROWS_TEXT)
        peptideFilePathLabel.setToolTip(Lang.CROSS.PEPTIDE_HELP)
        peptideFilePathLabel.setWhatsThis(Lang.CROSS.PEPTIDE_HELP)
        peptideFilePathEdit = QLineEdit()
        peptideFilePathEdit.setToolTip(Lang.CROSS.PEPTIDE_HELP)
        peptideFilePathEdit.setWhatsThis(Lang.CROSS.PEPTIDE_HELP)
        peptideFilePathEdit.setReadOnly(True)

        previewPeptideTableLabel = QLabel()
        previewPeptideTableLabel.setText(Lang.CROSS.PREVIEW_PEPTIDE_TITLE)
        previewPeptideTableLabel.setToolTip(Lang.CROSS.PREVIEW_PEPTIDE_HELP)
        previewPeptideTableLabel.setWhatsThis(Lang.CROSS.PREVIEW_PEPTIDE_HELP)
        filePeptidePreviewTableWidget = QTableWidget(10, 1)
        filePeptidePreviewTableWidget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        filePeptidePreviewTableWidget.setToolTip(Lang.CROSS.PREVIEW_PEPTIDE_HELP)
        filePeptidePreviewTableWidget.setWhatsThis(Lang.CROSS.PREVIEW_PEPTIDE_HELP)
        filePeptidePreviewTableWidget.setItem(0, 0, QTableWidgetItem("Peptide"))
        filePeptidePreviewTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        def validateAndPreviewPeptideFile(fileURL: str):
            MHCToolOption.PEPTIDE_COLUMN_NAME = None
            with open(fileURL, 'r') as fileHandler:
                sep = ',' if os.path.splitext(fileURL)[1].lower() == '.csv' else '\t'
                columnNames = set([tok.strip() for tok in fileHandler.readline().split(sep)])
                peptide_column_names = {"peptide", "Peptide", "PEPTIDE"}.intersection(columnNames)
                if len(peptide_column_names) > 0:
                    MHCToolOption.PEPTIDE_COLUMN_NAME = peptide_column_names.pop()
            # let's preview
            isHasColumnName = None if (MHCToolOption.PEPTIDE_COLUMN_NAME == None) else 0
            preview_df: pd.DataFrame = pd.read_csv(fileURL, sep, header=isHasColumnName, nrows=10, skipinitialspace=True)
            filePeptidePreviewTableWidget.clearContents()
            filePeptidePreviewTableWidget.setItem(0, 0, QTableWidgetItem("Peptide"))
            for ir, (_ir, row) in enumerate(preview_df.iterrows()):
                filePeptidePreviewTableWidget.setItem(ir + 1, 0, QTableWidgetItem(row[MHCToolOption.PEPTIDE_COLUMN_NAME or 0]))

            peptideFilePathEdit.setText(fileURL)
            MHCToolOption.PEPTIDE_PATH = fileURL

        def browsFilePeptide(_mouseEvent: QMouseEvent):
            if not peptideFilePathEdit.underMouse():
                return
            dialog = QFileDialog()
            dialog.setWindowTitle(Lang.CROSS.PEPTIDE_BROWS_TEXT)
            # dialog.setDefaultSuffix('csv')
            dialog.setNameFilters(['All files(*)', 'Text Doctuments (*.txt)', 'CSV (*.csv)', 'TSV (*.tsv)'])
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                fileURL = dialog.selectedFiles()[0]
                validateAndPreviewPeptideFile(fileURL)

        peptideFilePathEdit.mouseReleaseEvent = browsFilePeptide

        # allele
        alleleFilePathLabel = QLabel()
        alleleFilePathLabel.setText(Lang.CROSS.ALLELE_BROWS_TEXT)
        alleleFilePathLabel.setToolTip(Lang.CROSS.ALLELE_HELP)
        alleleFilePathLabel.setWhatsThis(Lang.CROSS.ALLELE_HELP)
        alleleFilePathEdit = QLineEdit()
        alleleFilePathEdit.setToolTip(Lang.CROSS.ALLELE_HELP)
        alleleFilePathEdit.setWhatsThis(Lang.CROSS.ALLELE_HELP)
        alleleFilePathEdit.setReadOnly(True)
        previewAlleleTableLabel = QLabel()
        previewAlleleTableLabel.setText(Lang.CROSS.PREVIEW_ALLELE_TITLE)
        previewAlleleTableLabel.setToolTip(Lang.CROSS.PREVIEW_ALLELE_HELP)
        previewAlleleTableLabel.setWhatsThis(Lang.CROSS.PREVIEW_ALLELE_HELP)
        fileAllelePreviewTableWidget = QTableWidget(10, 1)
        fileAllelePreviewTableWidget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        fileAllelePreviewTableWidget.setToolTip(Lang.CROSS.PREVIEW_ALLELE_HELP)
        fileAllelePreviewTableWidget.setWhatsThis(Lang.CROSS.PREVIEW_ALLELE_HELP)
        fileAllelePreviewTableWidget.setItem(0, 0, QTableWidgetItem("Allele"))
        fileAllelePreviewTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        def validateAndPreviewAlleleFile(fileURL: str):
            MHCToolOption.ALLELE_COLUMN_NAME = None
            with open(fileURL, 'r') as fileHandler:
                sep = ',' if os.path.splitext(fileURL)[1].lower() == '.csv' else '\t'
                columnNames = set([tok.strip() for tok in fileHandler.readline().split(sep)])
                allele_column_names = {"allele", "Allele", "ALLELE"}.intersection(columnNames)
                if len(allele_column_names) > 0:
                    MHCToolOption.ALLELE_COLUMN_NAME = allele_column_names.pop()
            # let's preview
            isHasColumnName = None if (MHCToolOption.ALLELE_COLUMN_NAME == None) else 0
            preview_df: pd.DataFrame = pd.read_csv(fileURL, sep, header=isHasColumnName, nrows=10, skipinitialspace=True)
            fileAllelePreviewTableWidget.clearContents()
            fileAllelePreviewTableWidget.setItem(0, 0, QTableWidgetItem("Allele"))
            for ir, (_ir, row) in enumerate(preview_df.iterrows()):
                fileAllelePreviewTableWidget.setItem(ir + 1, 0, QTableWidgetItem(row[MHCToolOption.ALLELE_COLUMN_NAME or 0]))

            alleleFilePathEdit.setText(fileURL)
            MHCToolOption.ALLELE_PATH = fileURL

        def browsFileAllele(_mouseEvent: QMouseEvent):
            if not alleleFilePathEdit.underMouse():
                return
            dialog = QFileDialog()
            # dialog.setDefaultSuffix('csv')
            dialog.setWindowTitle(Lang.CROSS.ALLELE_BROWS_TEXT)
            dialog.setNameFilters(['All files(*)', 'Text Doctuments (*.txt)', 'CSV (*.csv)', 'TSV (*.tsv)'])
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                fileURL = dialog.selectedFiles()[0]
                validateAndPreviewAlleleFile(fileURL)

        alleleFilePathEdit.mouseReleaseEvent = browsFileAllele

        multipleFileModeTab = QWidget()

        crossFilePathLayout = QGridLayout()
        crossFilePathLayout.addWidget(peptideFilePathLabel, 0, 0, 1, 1)
        crossFilePathLayout.addWidget(alleleFilePathLabel, 0, 1, 1, 1)
        crossFilePathLayout.addWidget(peptideFilePathEdit, 1, 0, 1, 1)
        crossFilePathLayout.addWidget(alleleFilePathEdit, 1, 1, 1, 1)
        crossFilePathLayout.addWidget(previewPeptideTableLabel, 2, 0, 1, 1)
        crossFilePathLayout.addWidget(filePeptidePreviewTableWidget, 3, 0, 1, 1)
        crossFilePathLayout.addWidget(previewAlleleTableLabel, 2, 1, 1, 1)
        crossFilePathLayout.addWidget(fileAllelePreviewTableWidget, 3, 1, 1, 1)
        multipleFileModeTab.setLayout(crossFilePathLayout)

        def switchTab(index: int):
            MHCToolOption.MODE = ('CSV', 'CROSS',)[index]
            if MHCToolOption.MODE == 'CSV' and len(filePathEdit.text()) > 0:
                validateAndPreviewCSVFile(filePathEdit.text())
            elif MHCToolOption.MODE == 'CROSS':
                if len(peptideFilePathEdit.text()) > 0:
                    validateAndPreviewPeptideFile(peptideFilePathEdit.text())
                if len(alleleFilePathEdit.text()) > 0:
                    validateAndPreviewAlleleFile(alleleFilePathEdit.text())
        self.inputTabWidget.addTab(pairModeTab, Lang.CSV.LABEL)
        self.inputTabWidget.addTab(multipleFileModeTab, Lang.CROSS.LABEL)
        self.inputTabWidget.currentChanged.connect(switchTab)
        self.inputTabWidget.setWhatsThis(Lang.MODE)

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        # self.progressBar.setValue(0)
        self.progressBar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # timer = QTimer(self)
        # timer.timeout.connect(self.advanceProgressBar)
        # timer.start(1000)

    def createRunConfigGroupBox(self):
        self.runConfigGroupBox = QGroupBox(Lang.RUNCONFIG.LABEL)
        self.runConfigGroupBox.setToolTip(Lang.RUNCONFIG.HELP)
        self.runConfigGroupBox.setWhatsThis(Lang.RUNCONFIG.HELP)

        def setIgnoreUnknowCheckBox(checked: bool):
            MHCToolOption.IGNORE_UNKNOW = checked

        def setUseEnsembleCheckBox(checked: bool):
            MHCToolOption.USE_ENSEMBLE = checked

        def selectLocDir(_mouseEvent: QMouseEvent):
            if not self.selectLogLocPathEdit.underMouse():
                return
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.FileMode.AnyFile)
            dialog.setWindowTitle(Lang.RUNCONFIG.LOG_DIR_TITLE)
            dialog.selectFile(MHCToolOption.LOG_UNKNOW_PATH)
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                fileURL = dialog.selectedFiles()[0]
                MHCToolOption.LOG_UNKNOW_PATH = fileURL
            tooltipText = f"{Lang.RUNCONFIG.LOG_DIR_HELP} '{MHCToolOption.LOG_UNKNOW_PATH}'"
            # self.selectLogLocPathEdit.setText(os.path.basename(MHCToolOption.LOG_UNKNOW_PATH))
            self.selectLogLocPathEdit.setText(MHCToolOption.LOG_UNKNOW_PATH)
            self.logUnknowCheckBox.setToolTip(tooltipText)
            self.selectLogLocPathEdit.setToolTip(tooltipText)

        def setLogUnknowCheckBox(checked: bool):
            MHCToolOption.LOG_UNKNOW = checked
            self.selectLogLocPathEdit.setEnabled(checked)

        self.ignoreUnknowCheckBox = QCheckBox(Lang.RUNCONFIG.IGNORE_UNK_LABEL)
        self.ignoreUnknowCheckBox.toggled.connect(setIgnoreUnknowCheckBox)
        self.ignoreUnknowCheckBox.setChecked(MHCToolOption.IGNORE_UNKNOW)
        self.ignoreUnknowCheckBox.setWhatsThis(Lang.RUNCONFIG.IGNORE_UNK_HELP)
        self.ignoreUnknowCheckBox.setToolTip(Lang.RUNCONFIG.IGNORE_UNK_HELP)

        self.useEnsembleCheckBox = QCheckBox(Lang.RUNCONFIG.ENSEMBLE_LABEL)
        self.useEnsembleCheckBox.toggled.connect(setUseEnsembleCheckBox)
        self.useEnsembleCheckBox.setChecked(MHCToolOption.USE_ENSEMBLE)
        self.useEnsembleCheckBox.setWhatsThis(Lang.RUNCONFIG.ENSEMBLE_HELP)
        self.useEnsembleCheckBox.setToolTip(Lang.RUNCONFIG.ENSEMBLE_HELP)

        tooltipText = f"{Lang.RUNCONFIG.LOG_DIR_HELP} '{MHCToolOption.LOG_UNKNOW_PATH}'"
        self.logUnknowCheckBox = QCheckBox(Lang.RUNCONFIG.LOG_DIR_LABEL)
        self.logUnknowCheckBox.toggled.connect(setLogUnknowCheckBox)
        self.logUnknowCheckBox.setChecked(MHCToolOption.LOG_UNKNOW)
        self.logUnknowCheckBox.setToolTip(tooltipText)
        self.logUnknowCheckBox.setWhatsThis(tooltipText)

        self.selectLogLocPathEdit = QLineEdit(MHCToolOption.LOG_UNKNOW_PATH)
        self.selectLogLocPathEdit.setToolTip(tooltipText)
        self.selectLogLocPathEdit.setWhatsThis(tooltipText)
        self.selectLogLocPathEdit.setEnabled(MHCToolOption.LOG_UNKNOW)
        self.selectLogLocPathEdit.setReadOnly(True)
        self.selectLogLocPathEdit.mouseReleaseEvent = selectLocDir

        def selectGPUHandler(index: int):
            MHCToolOption.GPU_ID = index - 1

        useGPULabel = QLabel()
        useGPULabel.setText(Lang.RUNCONFIG.GPU_LABEL)
        useGPULabel.setToolTip(Lang.RUNCONFIG.GPU_HELP)
        useGPULabel.setWhatsThis(Lang.RUNCONFIG.GPU_HELP)
        self.gpuDropDown = QComboBox()
        self.gpuDropDown.addItems([Lang.RUNCONFIG.NO_GPU] + [gpuDevice.name for gpuDevice in GPU_PHYSICAL_DEVICES])
        self.gpuDropDown.setEnabled(len(GPU_PHYSICAL_DEVICES) > 0)
        self.gpuDropDown.currentIndexChanged.connect(selectGPUHandler)
        self.gpuDropDown.setToolTip(Lang.RUNCONFIG.GPU_HELP)
        self.gpuDropDown.setWhatsThis(Lang.RUNCONFIG.GPU_HELP)

        rankEL = QCheckBox(Lang.RUNCONFIG.RANK_EL_LABEL)
        rankEL.setToolTip(Lang.RUNCONFIG.RANK_EL_HELP)
        rankEL.setWhatsThis(Lang.RUNCONFIG.RANK_EL_HELP)

        externalAlleleCheckBox = QCheckBox(Lang.RUNCONFIG.ADD_ALLELE)
        externalAlleleCheckBox.setToolTip(Lang.RUNCONFIG.ADD_ALLELE_HELP)
        externalAlleleCheckBox.setWhatsThis(Lang.RUNCONFIG.ADD_ALLELE_HELP)
        externalAllelePathEdit = QLineEdit(MHCToolOption.ALLELE_MAPPER_PATH)
        externalAllelePathEdit.setToolTip(Lang.RUNCONFIG.ADD_ALLELE_HELP)
        externalAllelePathEdit.setWhatsThis(Lang.RUNCONFIG.ADD_ALLELE_HELP)
        externalAllelePathEdit.setEnabled(False)
        externalAllelePathEdit.setReadOnly(True)

        def handleEnableAdditionalAllele(checked: bool):
            externalAllelePathEdit.setEnabled(checked)
            if not checked:
                MHCToolOption.ALLELE_MAPPER_PATH = "resources/allele_mapper"
                externalAllelePathEdit.setText(MHCToolOption.ALLELE_MAPPER_PATH)

        externalAlleleCheckBox.toggled.connect(handleEnableAdditionalAllele)

        def browsAdditionalAlleleFolder(_mouseEvent: QMouseEvent):
            if not externalAllelePathEdit.underMouse():
                return
            dialog = QFileDialog()
            dialog.setWindowTitle(Lang.RUNCONFIG.ADD_ALLELE_BROWS_TEXT)
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            dialog.selectFile(os.path.abspath(MHCToolOption.ALLELE_MAPPER_PATH))

            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                fileURL = dialog.selectedFiles()[0]
                MHCToolOption.ALLELE_MAPPER_PATH = fileURL
                externalAllelePathEdit.setText(MHCToolOption.ALLELE_MAPPER_PATH)

        externalAllelePathEdit.mouseReleaseEvent = browsAdditionalAlleleFolder

        outputLabel = QLabel(Lang.RUNCONFIG.OUTPUT_LABEL)
        outputLabel.setToolTip(Lang.RUNCONFIG.OUTPUT_HELP)
        outputLabel.setWhatsThis(Lang.RUNCONFIG.OUTPUT_HELP)
        outputPathEdit = QLineEdit(MHCToolOption.OUTPUT_DIRECTORY)
        outputPathEdit.setToolTip(Lang.RUNCONFIG.OUTPUT_HELP)
        outputPathEdit.setWhatsThis(Lang.RUNCONFIG.OUTPUT_HELP)
        outputPathEdit.setReadOnly(True)

        def selectOutputDir(_mouseEvent: QMouseEvent):
            if not outputPathEdit.underMouse():
                return
            dialog = QFileDialog()
            dialog.setNameFilters(['CSV (*.csv)', 'TSV (*.tsv)'])
            dialog.setWindowTitle(Lang.RUNCONFIG.OUTPUT_LABEL)
            dialog.selectFile(MHCToolOption.OUTPUT_DIRECTORY)
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                fileURL = dialog.selectedFiles()[0]
                MHCToolOption.OUTPUT_DIRECTORY = fileURL
                outputPathEdit.setText(MHCToolOption.OUTPUT_DIRECTORY)

        outputPathEdit.mouseReleaseEvent = selectOutputDir

        layout = QGridLayout()
        layout.addWidget(self.ignoreUnknowCheckBox, 0, 0, 1, 1)
        layout.addWidget(self.logUnknowCheckBox, 0, 1, 1, 1)
        layout.addWidget(useGPULabel, 0, 2, 1, 2)
        layout.addWidget(self.useEnsembleCheckBox, 1, 0, 1, 1)
        layout.addWidget(self.selectLogLocPathEdit, 1, 1, 1, 1)
        layout.addWidget(self.gpuDropDown, 1, 2, 1, 2)
        # layout.addLayout(externalAlleleLayout, 2, 0, 1, 3)
        layout.addWidget(rankEL, 2, 0, 1, 1)
        layout.addWidget(externalAlleleCheckBox, 2, 1, 1, 1)
        layout.addWidget(externalAllelePathEdit, 3, 1, 1, 1)
        layout.addWidget(outputLabel, 2, 2, 1, 1)
        layout.addWidget(outputPathEdit, 3, 2, 1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 2)
        self.runConfigGroupBox.setLayout(layout)


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    guiWindow = MHCToolGUI()
    guiWindow.show()
    sys.exit(app.exec())
