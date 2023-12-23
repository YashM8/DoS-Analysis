"""
Run this file for the DoS Analysis Tool.

The automatic file analyzer runs first. After it is done, it stops, the window closes and the manual verification
app runs.
"""

from processing import *
from verify import *

if __name__ == "__main__":
    """
    Input Parameters - 
    ====================================================================================================================
    """
    framesPerSecond = 2999
    needleWidth = 2.11                            # in millimeter
    rowsToSkip = 0                                # no more than 30 for accurate results.
    folder = "/Users/ypm/Desktop/CurrProjects/DoS-Analysis/videos/difffps"
    showVidAndPlot = False                        # True to show the video and plots while being processed
    breakpoints = 6                               # Keep between 5 - 7. Ineffective above 10 and will take much longer.
    """
    ====================================================================================================================
    """

    dos_app = DosAnalyzerApp(directory=folder,
                             fps=framesPerSecond,
                             needle_width=needleWidth,
                             skip=(rowsToSkip + 1),
                             show_the_vid=showVidAndPlot,
                             breaks=breakpoints)
    dos_app.run()

    user_response = messagebox.askyesno("Verification", "Do you want to verify?")

    if user_response:
        verify_app = VerifierApp()
        verify_app.run()
