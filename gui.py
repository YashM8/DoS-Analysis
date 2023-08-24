import os
import tkinter as tk
from tkinter import filedialog
from analyze import *
from measure import *
from tkinter import messagebox

# Global variable for the directory
folder = None


def gui_dos():
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

        # Iterate through the list of .mp4 files
        for index, mp4_file in enumerate(mp4_files):
            # Call the function 'measureWidths' with specific parameters and store the returned DataFrame in 'df'
            df = measureWidths(filename=mp4_file, needle_mm=needle, fps=frames_ps, show=False, skip=skip_cols)

            # Call the function 'autoRegressor' on the DataFrame 'df', which returns a plot and a slope value
            # plot, slope, orig_slopes = autoRegressor(df)
            # start, stop, orig = findRegion(df, partitions=30, threshold=8)
            # slope, plot = findSlope(orig, start, stop)

            myPlot, slope, all_5_slopes = piecewise(df)

            # Create a directory based on the name of the .mp4 file (without the extension) to store plot and CSV
            plot_directory = os.path.splitext(mp4_file)[0]
            # Create the directory if it doesn't exist
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

        # Save the 'slope_data' and 'troubleshoot_data'
        result_csv_filename = os.path.join(directory, 'SLOPE_DATA.csv')
        troubleshoot_filename = os.path.join(directory, 'TROUBLESHOOT.csv')

        # Save the data as CSV files
        pd.DataFrame(slope_data).to_csv(result_csv_filename, index=False)
        pd.DataFrame(troubleshoot_data).to_csv(troubleshoot_filename, index=False)

    def analyze_files():
        """
        Command to the 'Analyze' button.

        :return: None
        """
        # Convert the value entered in the fps_entry widget to an integer and store it in the 'fps' variable
        fps = int(fps_entry.get())

        # Convert the value entered in the needle_width_entry to a floating-point number
        needle_w = float(needle_width_entry.get())

        # Convert the value chosen in the slider widget to an integer and store it in the 'skip' variable
        skip = int(slider.get())

        # Check if a directory 'folder' has been selected
        if folder is not None:

            # Call the 'process_mp4_file' function with the provided parameters
            process_mp4_file(directory=folder, needle=needle_w, frames_ps=fps, skip_cols=skip)

        else:
            # If no directory has been selected, show an error message box
            messagebox.showerror("Error", "Please choose directory!")

        # Clear the contents of the fps_entry widget
        fps_entry.delete(0, tk.END)

        # Clear the contents of the needle_width_entry widget
        needle_width_entry.delete(0, tk.END)

        root.destroy()

    def browse_directory():
        """
        Command to 'Browse Directory' button.

        :return: None
        """
        # Declare the 'folder' variable as global, so it can be accessed and modified within functions
        global folder

        # Open a directory selection dialog box using the 'filedialog.askdirectory' function
        directory = filedialog.askdirectory(initialdir="Desktop", title="Select Directory")

        # Assign the selected directory to the global variable 'folder'
        folder = directory

    # Create the main application window
    root = tk.Tk()
    root.title("DoS Data Analysis")

    # Create a frame for input elements
    input_frame = tk.Frame(root)
    # input_frame.pack(side="left", padx=20, pady=20)
    input_frame.grid(row=0, column=0)

    # Create labels and entry fields for FPS and needle width
    fps_label = tk.Label(input_frame, text="FPS:")
    fps_label.grid(row=1, column=0, sticky="w", padx=10)

    fps_entry = tk.Entry(input_frame)
    fps_entry.grid(row=1, column=1, padx=5)
    fps_entry.insert(0, '2999')

    needle_width_label = tk.Label(input_frame, text="Needle Width:")
    needle_width_label.grid(row=2, column=0, sticky="w", padx=5)

    needle_width_entry = tk.Entry(input_frame)
    needle_width_entry.grid(row=2, column=1, padx=5)
    needle_width_entry.insert(0, '2.11')

    # Create a button to select a directory
    directory_button = tk.Button(input_frame, text="Select Folder", command=browse_directory)
    directory_button.grid(row=1, columnspan=2, pady=10, column=3, padx=10)

    # Create a button to trigger file analysis
    analyze_button = tk.Button(input_frame, text="Analyze Files", command=analyze_files)
    analyze_button.grid(row=2, columnspan=2, pady=10, column=3, padx=10)

    # Create a slider control
    slider = tk.Scale(root, from_=50, to=1, orient="vertical", length=150)
    slider.grid(row=0, column=4, padx=30, pady=10)

    # Create a label to signal the end of the program
    starting_label = tk.Label(root, text="", font=("Courier", 20, "bold"))
    starting_label.grid(row=4, padx=10, pady=10)

    # Start the main event loop to display the GUI
    root.mainloop()
