# Import necessary modules
import os
from tkinter import messagebox
from analyze import *
from measure import *


# Define a function for processing MP4 files in a directory.
def process_mp4_file(directory, needle, frames_ps, skip_cols, show_or_not):
    # Create an empty list to store MP4 file paths.
    mp4_files = []

    # Traverse through the specified directory and subdirectories to find MP4 files.
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))

    # If no MP4 files are found, show an error message and exit the function.
    if len(mp4_files) == 0:
        messagebox.showerror("Error", "This directory has no MP4 files!\n\nPlease start the process again.")
        return

    # Initialize empty lists to store slope and troubleshooting data.
    slope_data = []
    troubleshoot_data = []
    print("\nStarting...\n")

    # Iterate through each MP4 file and analyze it.
    for index, mp4_file in enumerate(mp4_files):
        try:
            # Measure widths using the 'measureWidths' function from the 'measure' module.
            df = measureWidths(filename=mp4_file, needle_mm=needle, fps=frames_ps, show=show_or_not, skip=skip_cols)

            # Perform piecewise analysis and get plot data.
            myPlot, slope, all_5_slopes = piecewise(df)

            # Extract the file name from the path.
            name = os.path.basename(mp4_file)

            # Create a directory for saving analysis results.
            plot_directory = os.path.splitext(mp4_file)[0]
            os.makedirs(plot_directory, exist_ok=True)

            # Save the data to CSV file.
            csv_filename = os.path.join(plot_directory, 'OriginalData.csv')
            df.to_csv(csv_filename, index=False)

            # Save the plot as an image file.
            plot_filename = os.path.join(plot_directory, 'OverlayPlot.png')
            myPlot.savefig(plot_filename, dpi=200)
            myPlot.close()

            # Append slope and relaxation time data to 'slope_data' list.
            slope_data.append({'Filename': name,
                               'Slope': slope,
                               'Relaxation Time': -1 / (3 * slope)
                               })

            # Append troubleshooting data to 'troubleshoot_data' list.
            troubleshoot_data.append({'Filename': name,
                                      'Slope': slope,
                                      'All Slopes': all_5_slopes,
                                      })

            # Print a message indicating successful file processing.
            print(f"File Processed - {name}\n")

        except Exception as e:
            # Display an error message if there is an exception during file processing.
            messagebox.showerror("Error", f"Error analyzing file: {e}")

    # After processing all files, print the total number of files processed.
    print(f"{len(mp4_files)} Files Processed. Flagging Bad Data...\n")

    count = flagFiles(directory)

    print(f"{count} Files Processed.\n")
    print("DONE! Find flagged files with the folder name beginning with '0_FLAG_...'\n")

    # Define output file paths for slope and troubleshooting data CSV files.
    result_csv_filename = os.path.join(directory, 'SLOPE_DATA.csv')
    troubleshoot_filename = os.path.join(directory, 'TROUBLESHOOT.csv')

    # Save slope and troubleshooting data to CSV files.
    pd.DataFrame(slope_data).to_csv(result_csv_filename, index=False)
    pd.DataFrame(troubleshoot_data).to_csv(troubleshoot_filename, index=False)


# Define a class for the DOS Analyzer application.
class DosAnalyzerApp:
    def __init__(self, directory, needle_width, fps, skip, show_the_vid):
        self.directory = directory
        self.needle_width = needle_width
        self.fps = fps
        self.skip = skip
        self.show = show_the_vid

    # Method for analyzing files based on user inputs.
    def analyze_files(self):
        try:
            # Check if user inputs are valid numeric values.
            assert type(self.needle_width) == float
            assert type(self.fps) == int
            assert type(self.skip) == int
            assert type(self.directory) == str
        except ValueError:
            # Display an error message if the inputs are not valid.
            messagebox.showerror("Error", "Invalid input. Please enter valid numeric values.")
            return

        # Check if a directory is selected.
        if self.directory is not None:
            # Call the 'process_mp4_file' function to start the analysis.
            process_mp4_file(directory=self.directory, needle=self.needle_width,
                             frames_ps=self.fps, skip_cols=self.skip, show_or_not=self.show)
        else:
            # Display an error message if no directory is selected.
            messagebox.showerror("Error", "Please choose a directory!")

    # Method to run the DOS Analyzer application.
    def run(self):
        # Call the 'analyze_files' method to perform the analysis.
        self.analyze_files()


if __name__ == "__main__":

    framesPerSecond = 2999
    needleWidth = 2.11  # in mm
    rowsToSkip = 0  # not more than 30
    folder = "/Users/ypm/Desktop/DoS/test files"  # CHANGE THIS
    showVidAndPlot = False

    """
    ===========================================================================
    """

    dos_app = DosAnalyzerApp(directory=folder,
                             fps=framesPerSecond,
                             needle_width=needleWidth,
                             skip=(rowsToSkip + 1),
                             show_the_vid=showVidAndPlot)
    dos_app.run()
