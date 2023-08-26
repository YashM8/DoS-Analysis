"""
Run this file to show the GUI for the DoS Analysis Tool.
"""

from gui import gui_dos
from  verify import VerifierApp

if __name__ == "__main__":
    gui_dos()
    app = VerifierApp()
    app.main()

