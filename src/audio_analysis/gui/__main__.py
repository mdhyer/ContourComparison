from __future__ import annotations

import sys
from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def run_gui():
    """Entry point for the Audio Analysis GUI."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()