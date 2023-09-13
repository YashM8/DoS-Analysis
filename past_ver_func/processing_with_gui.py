import os
# import tkinter as tk
# from tkinter import filedialog
from tkinter import messagebox
# from tkinter import ttk
# from ttkthemes import ThemedTk

from analyze import *
from measure import *


def process_mp4_file(directory, needle, frames_ps, skip_cols):
    """
    Processes every MP4 file in this directory.

    :param skip_cols: Number of columns to skip.
    :param frames_ps: Original FPS of the video.
    :param needle: Needle width in millimeter.
    :param directory: Directory to process.
    :return: None
    """
    # Initialize an empty list to store the paths of found .mp4 files
    mp4_files = []

    # Traverse through the directory and its subdirectories using os.walk
    for root, _, files in os.walk(directory):
        # Iterate through the files in the current directory
        for file in files:
            # Check if the file has a .mp4 extension (case-insensitive)
            if file.lower().endswith(".mp4"):
                # If the condition is met, add the complete path of the .mp4 file to the list
                mp4_files.append(os.path.join(root, file))

    # Check if the directory has mp4 files in it
    if len(mp4_files) == 0:
        messagebox.showerror("Error", "This directory has no MP4 files!\n\nPlease start process again.")

    # Initialize an empty list to store slope-related data
    slope_data = []
    troubleshoot_data = []
    print("Starting...")

    # Iterate through the list of .mp4 files
    for index, mp4_file in enumerate(mp4_files):
        # Call the function 'measureWidths' with specific parameters and store the returned DataFrame in 'df'
        try:
            df = measureWidths(filename=mp4_file, needle_mm=needle, fps=frames_ps, show=False, skip=skip_cols)
            # Call the function 'autoRegressor' on the DataFrame 'df', which returns a plot and a slope value
            # plot, slope, orig_slopes = autoRegressor(df)
            # start, stop, orig = findRegion(df, partitions=30, threshold=8)
            # slope, plot = findSlope(orig, start, stop)

            myPlot, slope, all_5_slopes = piecewise(df)

            # Create a directory based on the name of the .mp4 file (without the extension) to store plot and CSV
            plot_directory = os.path.splitext(mp4_file)[0]
            os.makedirs(plot_directory, exist_ok=True)

            # Create a CSV file with the original data and save the DataFrame 'df' to it
            csv_filename = os.path.join(plot_directory, 'OriginalData.csv')
            df.to_csv(csv_filename, index=False)

            # Create a filename for the plot and save the plot to an image file
            plot_filename = os.path.join(plot_directory, 'OverlayPlot.png')
            myPlot.savefig(plot_filename, dpi=200)
            myPlot.close()  # Close the plot to free up resources

            # split_info = os.path.basename(mp4_file).split()
            # con = split_info[0]
            # compound = split_info[1] + split_info[2]

            # Append slope-related data for this .mp4 file to the 'slope_data' list
            # slope_data.append({'Compound': compound,
            #                    'Concentration': con,
            #                    'Slope': slope,
            #                    'Relaxation Time': 1 / (3 * slope)
            #                    })

            name = os.path.basename(mp4_file)

            slope_data.append({'Filename': name,
                               'Slope': slope,
                               'Relaxation Time': -1 / (3 * slope)
                               })

            troubleshoot_data.append({'Filename': name,
                                      'Slope': slope,
                                      'All Slopes': all_5_slopes,
                                      })

            # Insert a log message indicating the processing of the current file into a text widget or similar
            print(f"File Processed - {name}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing file: {e}")

    print(f"{len(mp4_files)} Files Processed.")

    # Save the 'slope_data' and 'troubleshoot_data'
    result_csv_filename = os.path.join(directory, 'SLOPE_DATA.csv')
    troubleshoot_filename = os.path.join(directory, 'TROUBLESHOOT.csv')

    # Save the data as CSV files
    pd.DataFrame(slope_data).to_csv(result_csv_filename, index=False)
    pd.DataFrame(troubleshoot_data).to_csv(troubleshoot_filename, index=False)


class DosAnalyzerApp:
    def __init__(self, directory, needle_width, fps, skip):
        # Initialize instance variables for GUI elements and default values
        # self.starting_label = None
        # self.slider = None
        # self.analyze_button = None
        # self.directory_button = None
        # self.needle_width_entry = None
        # self.fps_entry = None
        # self.folder = None
        # self.root = tk.Tk()
        # self.root.title("DoS Data Analysis")

        # self.setup_gui()
        self.directory = directory
        self.needle_width = needle_width
        self.fps = fps
        self.skip = skip

    # def setup_gui(self):
    #     # Set up the graphical user interface (GUI) elements
    #
    #     # Create a frame to hold input widgets
    #     input_frame = tk.Frame(self.root)
    #     input_frame.grid(row=0, column=0)
    #
    #     # Create and place labels and entry fields for FPS and Needle Width
    #     fps_label = tk.Label(input_frame, text="FPS:")
    #     fps_label.grid(row=1, column=0, sticky="w", padx=10)
    #
    #     self.fps_entry = tk.Entry(input_frame)
    #     self.fps_entry.grid(row=1, column=1, padx=5)
    #     self.fps_entry.insert(0, '2999')  # Set default FPS value
    #
    #     needle_width_label = tk.Label(input_frame, text="Needle Width:")
    #     needle_width_label.grid(row=2, column=0, sticky="w", padx=5)
    #
    #     self.needle_width_entry = tk.Entry(input_frame)
    #     self.needle_width_entry.grid(row=2, column=1, padx=5)
    #     self.needle_width_entry.insert(0, '2.11')  # Set default needle width value
    #
    #     # Create buttons to select a folder and analyze files
    #     self.directory_button = tk.Button(input_frame, text="Select Folder", command=self.browse_directory)
    #     self.directory_button.grid(row=1, columnspan=2, pady=10, column=3, padx=10)
    #
    #     self.analyze_button = tk.Button(input_frame, text="Analyze Files", command=self.analyze_files)
    #     self.analyze_button.grid(row=2, columnspan=2, pady=10, column=3, padx=10)
    #
    #     # Create a slider for skip columns and a label to show current value
    #     self.slider = tk.Scale(self.root, from_=50, to=1, orient="vertical", length=150)
    #     self.slider.grid(row=0, column=4, padx=30, pady=10)
    #
    #     # Create a label to display analysis status
    #     self.starting_label = tk.Label(self.root, text="", font=("Courier", 20, "bold"))
    #     self.starting_label.grid(row=4, padx=10, pady=10)

    def analyze_files(self):
        # Function to analyze files with user-provided parameters

        try:
            # Extract user-provided values for analysis
            # self.fps = int(self.fps_entry.get())
            # self.needle_width = float(self.needle_width_entry.get())
            # self.skip = int(self.slider.get())

            assert type(self.needle_width) == int
            assert type(self.fps) == int
            assert type(self.skip) == int
            assert type(self.needle_width) == str

        except ValueError:
            # Handle invalid input with an error message
            messagebox.showerror("Error", "Invalid input. Please enter valid numeric values.")
            return

        if self.directory is not None:
            # If a folder is selected, perform analysis using provided parameters
            process_mp4_file(directory=self.directory, needle=self.needle_width,
                             frames_ps=self.fps, skip_cols=self.skip)
        else:
            # If no folder is selected, show an error message
            messagebox.showerror("Error", "Please choose a directory!")

    # def browse_directory(self):
    #     # Function to open a dialog for selecting a directory
    #
    #     try:
    #         # Open a dialog for selecting a directory and store the chosen path
    #         self.folder = filedialog.askdirectory(initialdir="Desktop", title="Select Directory")
    #     except AttributeError:
    #         # Handle cases where folder selection was canceled
    #         messagebox.showerror("Error", "Folder selection was canceled.")

    def run(self):
        self.analyze_files()
