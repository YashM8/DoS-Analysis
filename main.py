"""
Run this file to show the GUI for the DoS Analysis Tool.

The automatic file analyzer runs first. After it is done, it stops, the window closes and the manual verification
app runs
"""

from gui import *
from verify import *

if __name__ == "__main__":
    dos_app = DosAnalyzerApp()
    dos_app.run()

    user_response = messagebox.askyesno("Verification", "Do you want to verify?")

    if user_response:
        verify_app = VerifierApp()
        verify_app.run()

