from GUI.mainwindow import MainWindow
import sys
import argparse
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon
sys.path.append('./GUI/')
sys.path.append('./gsi_classification/')


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Path to an input image',
                        required=False,
                        type=str)
    return parser


args = build_argparser().parse_args()


app = QApplication(sys.argv)
app.setWindowIcon(QIcon('GUI/icon.ico'))

window = MainWindow(inputFile=args.input)
window.show()

# Start the event loop.
app.exec_()
