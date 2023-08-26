import os
import tkinter as tk
from tkinter import PhotoImage, Label, Button, filedialog, Entry
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from analyze import linspaceSmoother


class PhotoViewerApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Review and Make Changes")

        self.image_paths = []
        self.current_index = 0
        self.data = None
        self.dir = None

        self.browse_button = Button(self.root, text="Browse", command=self.browse_directory)
        self.browse_button.pack()

        self.load_image()

        self.prev_button = Button(self.root, text="Previous", command=self.show_previous_image)
        self.prev_button.pack()

        self.next_button = Button(self.root, text="Next", command=self.show_next_image)
        self.next_button.pack()

        self.rework_button = Button(self.root, text="Rework", command=self.rework_data)
        self.rework_button.pack()

        self.start_time_entry = Entry(self.root)
        self.start_time_entry.pack()
        self.stop_time_entry = Entry(self.root)
        self.stop_time_entry.pack()

        self.fit_button = Button(self.root, text="Fit", command=self.fit_and_plot)
        self.fit_button.pack()

        self.save_button = Button(self.root, text="Save", command=self.save_slope)
        self.save_button.pack()

        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT)

    def browse_directory(self):
        selected_directory = filedialog.askdirectory(initialdir="Desktop",
                                                     title="Select Directory")
        if selected_directory:
            self.dir = selected_directory
            self.image_paths = self.get_image_paths(selected_directory)
            self.current_index = 0
            self.update_image()

    def get_image_paths(self, directory):
        image_paths = []
        for root_dir, _, files in os.walk(directory):
            if "OverlayPlot.png" in files:
                image_paths.append(os.path.join(root_dir, "OverlayPlot.png"))
        return image_paths

    def load_image(self):
        if self.image_paths and 0 <= self.current_index < len(self.image_paths):
            image_path = self.image_paths[self.current_index]
            image = Image.open(image_path)
            image = image.resize((600, 300), Image.BILINEAR)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label = Label(self.frame, image=self.photo)
            self.image_label.pack(side=tk.LEFT)

    def show_previous_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.update_image()

    def show_next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.update_image()

    def update_image(self):
        if hasattr(self, 'image_label'):
            self.image_label.destroy()
        self.load_image()

    def rework_data(self):
        data_path = os.path.join(os.path.dirname(self.image_paths[self.current_index]), "OriginalData.csv")
        if os.path.exists(data_path):
            self.data = pd.read_csv(data_path)
            self.ax.clear()
            self.ax.scatter(self.data['Times'], self.data['Width'], s=15)
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Width')
            self.ax.set_title('Original Data')
            self.canvas.draw()
        else:
            print("OriginalData.csv not found.")

    def fit_and_plot(self):
        if self.data is not None:
            self.ax.clear()
            self.ax.scatter(self.data['Times'], self.data['Width'], label='Original Data', s=15)

            self.data = linspaceSmoother(self.data)

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

            self.ax.scatter(self.data['Times'], self.data['Width'], label='Smoothed Data', s=15, color='black')
            self.ax.plot(subset_data['Times'], y_pred, color='red', label='Linear Regression')
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Width')
            self.ax.set_title('Original Data and Linear Regression')
            self.ax.legend()
            self.canvas.draw()
            self.slope = model.coef_[0]

    def save_slope(self):
        directory = self.dir
        slope = self.slope
        name = os.path.basename(os.path.dirname(self.image_paths[self.current_index])) + '.mp4'

        csv_file_path = os.path.join(directory, 'SLOPE_DATA.csv')
        df = pd.read_csv(csv_file_path)

        mask = df['Filename'] == name
        print(mask)
        df.loc[mask, 'Slope'] = slope
        df.loc[mask, 'Relaxation Time'] = -1 / (3 * slope)

        df.to_csv(csv_file_path, index=False)

    def main(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PhotoViewerApp()
    app.main()
