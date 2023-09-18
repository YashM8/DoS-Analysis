# Import necessary libraries
import os
import tkinter as tk
from tkinter import Label, Button, filedialog, Entry
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from analyze import linspaceSmoother
from matplotlib.widgets import RectangleSelector


# Function to get image paths within a directory
def get_image_paths(directory):
    image_paths = []
    try:
        # Walk through the directory and its subdirectories
        for root_dir, _, files in os.walk(directory):
            # Check if OverlayPlot.png exists in the files list
            if "OverlayPlot.png" in files:
                image_paths.append(os.path.join(root_dir, "OverlayPlot.png"))
    except Exception as e:
        print("Error while getting image paths:", e)
    return image_paths


# Main application class
class VerifierApp:
    def __init__(self):
        """
        Initialize the GUI.
        """
        # Initialize variables
        self.image_label = None
        self.slope = None
        self.photo = None

        # Create the main application window
        self.root = tk.Tk()
        self.root.title("Review and Make Changes")

        self.image_paths = []  # List to store image file paths
        self.current_index = 0  # Index of the currently displayed image
        self.data = None  # DataFrame to store data
        self.dir = None  # Directory selected by the user

        # Create a frame for both the plot and the image
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create and place GUI components
        self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_directory)
        self.browse_button.pack()  # Display the "Browse" button

        self.load_image()  # Load and display the initial image

        # Buttons for navigating through images
        self.prev_button = tk.Button(self.root, text="Previous", command=self.show_previous_image)
        self.prev_button.pack()  # Display the "Previous" button

        self.next_button = tk.Button(self.root, text="Next", command=self.show_next_image)
        self.next_button.pack()  # Display the "Next" button

        # Button to rework data
        self.rework_button = tk.Button(self.root, text="Rework", command=self.rework_data)
        self.rework_button.pack()  # Display the "Rework" button

        # Entry fields for specifying start and stop times
        self.start_label = tk.Label(self.root, text="Start:")
        self.start_label.pack()  # Display the "Start:" label

        self.start_time_entry = tk.Entry(self.root)
        self.start_time_entry.pack()  # Display the entry field for start time

        self.stop_label = tk.Label(self.root, text="Stop:")
        self.stop_label.pack()  # Display the "Stop:" label

        self.stop_time_entry = tk.Entry(self.root)
        self.stop_time_entry.pack()  # Display the entry field for stop time

        # Button to fit data and plot
        self.fit_button = tk.Button(self.root, text="Fit", command=self.fit_and_plot)
        self.fit_button.pack()  # Display the "Fit" button

        # Button to save slope data
        self.save_button = tk.Button(self.root, text="Save", command=self.save_slope)
        self.save_button.pack()  # Display the "Save" button

        # Label to display the file name
        self.file_name_label = tk.Label(self.root, text='      ')
        self.file_name_label.pack(padx=10, pady=10)  # Display the file name label
        self.file_name_label.config(font=("Helvetica", 14))

        # Checkbox for applying smoothing
        self.smooth_var = tk.BooleanVar()
        self.smooth_var.set(True)  # Set the initial state to True
        self.smooth = tk.Checkbutton(
            self.plot_frame,
            text="Apply Smoothing",
            variable=self.smooth_var,
            onvalue=True,
            offvalue=False,
            command=self.checkbox
        )
        self.smooth.select()  # Initially, the checkbox is selected
        self.smooth.pack()  # Display the checkbox

        # Create a figure and canvas for displaying plots
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.ax.grid()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=True)  # Display the plot at the top of the frame

        # Attach a click event to the canvas
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def on_canvas_click(self, event):
        """
        Handles the mouse click event.

        - Left click selects start time.
        - Right click selects stop time.

        :param event: The click event.
        :return: None
        """
        # Handle mouse click events on the plot canvas
        if event.inaxes == self.ax:
            if event.button == 1:  # Left mouse button
                self.start_time_entry.delete(0, tk.END)
                self.start_time_entry.insert(0, str(round(event.xdata, 2)))
            elif event.button == 3:  # Right mouse button
                self.stop_time_entry.delete(0, tk.END)
                self.stop_time_entry.insert(0, str(round(event.xdata, 2)))

    def checkbox(self):
        """
        Toggles the checkbox.

        :return: boolean value. True iff the checkbox is clicked.
        """
        # Handle checkbox state changes
        smooth_state = self.smooth_var.get()  # Get the current state (True or False)
        return smooth_state

    # Function to browse and select a directory
    def browse_directory(self):
        """
        Browses the directory using the OS GUI.

        :return: None
        """
        try:
            selected_directory = filedialog.askdirectory(initialdir="Desktop", title="Select Directory")
            if selected_directory:
                self.dir = selected_directory
                self.image_paths = get_image_paths(selected_directory)  # Get image paths in the selected directory
                self.current_index = 0  # Reset the current image index
                self.update_image()  # Update the displayed image
        except Exception as e:
            print("Error while browsing directory:", e)

    # Function to load and display an image
    def load_image(self):
        """
        Loads the images from the directory and displays it below the plot.

        :return: None
        """
        if self.image_paths and 0 <= self.current_index < len(self.image_paths):
            image_path = self.image_paths[self.current_index]
            image = Image.open(image_path)
            image = image.resize((600, 300))  # Resize the image
            self.photo = ImageTk.PhotoImage(image)

            # Create a label for the image and display it below the plot
            self.image_label = Label(self.plot_frame, image=self.photo)
            self.image_label.pack(side=tk.BOTTOM)  # Display the image at the bottom of the frame

            name = os.path.basename(os.path.dirname(image_path))
            self.file_name_label.config(text=name)  # Update the file name label

    def show_previous_image(self):
        """
        Goes to the previous image.

        :return: None
        """
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.update_image()  # Update the displayed image

    def show_next_image(self):
        """
        Goes to the next image.

        :return: None
        """
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.update_image()  # Update the displayed image

    def update_image(self):
        """
        Updates the image label.

        :return: None
        """
        if hasattr(self, 'image_label') and self.image_label is not None:
            self.image_label.destroy()  # Remove the previous image label
        self.load_image()  # Load and display the current image

    def rework_data(self):
        """
        Loads the data and plots it.

        :return: None
        """
        data_path = os.path.join(os.path.dirname(self.image_paths[self.current_index]), "OriginalData.csv")
        if os.path.exists(data_path):
            self.data = pd.read_csv(data_path)  # Read data from OriginalData.csv
            self.ax.clear()
            self.ax.scatter(self.data['Times'], self.data['Width'], s=15, color='green')
            self.ax.grid()
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Width')
            self.ax.set_title('Original Data')
            self.canvas.draw()  # Redraw the plot
        else:
            print("OriginalData.csv not found.")

    def fit_and_plot(self):
        """
        Plots data and fits it after taking user input.

        :return: None
        """
        if self.data is not None:
            self.ax.clear()
            data_path = os.path.join(os.path.dirname(self.image_paths[self.current_index]), "OriginalData.csv")
            self.data = pd.read_csv(data_path)  # Read data from OriginalData.csv
            self.ax.scatter(self.data['Times'], self.data['Width'], label='Original Data', s=15)

            self.checkbox()
            if self.checkbox():
                self.data = linspaceSmoother(self.data)  # Apply custom smoothing function

            if self.start_time_entry.get() == '' or self.stop_time_entry.get() == '':
                start_time = self.data['Times'].min()
                stop_time = self.data['Times'].max()
            else:
                start_time = float(self.start_time_entry.get())
                stop_time = float(self.stop_time_entry.get())

            subset_data = self.data[(self.data['Times'] >= start_time) & (self.data['Times'] <= stop_time)]

            x = subset_data[['Times']]
            y = subset_data['Width']
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)

            if self.smooth_var.get():
                self.ax.scatter(self.data['Times'], self.data['Width'], label='Smoothed Data', s=15, color='black')
            self.ax.plot(subset_data['Times'], y_pred, color='red', label='Linear Regression')
            self.ax.grid()
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Width')
            self.ax.set_title('Original Data and Linear Regression')
            self.ax.legend()
            self.canvas.draw()  # Redraw the plot
            self.slope = model.coef_[0]  # Store the slope coefficient

    def save_slope(self):
        """
        Updates the slope of the linear region in "SLOPE_DATA.csv"

        :return: None
        """
        directory = self.dir
        slope = self.slope
        name = os.path.basename(os.path.dirname(self.image_paths[self.current_index])) + '.mp4'

        if name.startswith("0_FLAG_"):
            modified_string = name.replace("0_FLAG_", "", 1)
            name = modified_string

        csv_file_path = os.path.join(directory, 'SLOPE_DATA.csv')
        df = pd.read_csv(csv_file_path)

        mask = df['Filename'] == name
        df.loc[mask, 'Slope'] = slope
        df.loc[mask, 'Relaxation Time'] = -1 / (3 * slope)

        df.to_csv(csv_file_path, index=False)  # Save the updated slope data to the CSV file

    def run(self):
        """
        Starts the mainloop
        """
        self.root.mainloop()  # Start the main GUI event loop


# Entry point of the program
if __name__ == "__main__":
    app = VerifierApp()  # Create an instance of the VerifierApp class
    app.run()  # Start the GUI application
