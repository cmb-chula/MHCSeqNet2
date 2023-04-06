# worker.py
from core.options.InferenceOption import MHCToolOption
import typing
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from subprocess import Popen, PIPE


class SubProcessWorker(QObject):
    """
    This class will spawn subprocess and wait for it to complete
    """
    finished = pyqtSignal(bool)
    # intReady = pyqtSignal(int)

    def __init__(self, parent: typing.Optional['QObject'] = None, command: str = "python mhctool.py") -> None:
        super().__init__(parent=parent)
        self.command = command
        self.output: bytes = b''
        self.error_output: bytes = b''

    @pyqtSlot()
    def procCounter(self):  # A slot takes no params
        print(MHCToolOption())
        try:
            commands_args = MHCToolOption.getArgumentsString()
            print("Using the following command")
            print(self.command, *commands_args, sep=" \\\n\t")
            subProcess = Popen(args=self.command.split(' ') + commands_args, stdin=None, stdout=PIPE, stderr=PIPE)
            # print(f"Spawned Process {subProcess}")
            output, error_output = subProcess.communicate()
            self.output = output
            self.error_output = error_output
        except Exception as ex:
            # print(f"An error occour {ex}")
            self.error_output = bytes(f'self:{ex}', encoding='utf-8')
            self.finished.emit(False)
            return
        finally:
            print(f"Finished Process")
            pass
        # self.intReady.emit(i)

        self.finished.emit(True)
