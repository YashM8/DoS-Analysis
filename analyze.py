import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pwlf
from scipy.spatial import distance
import os
import fnmatch
import shutil
from sklearn.preprocessing import StandardScaler


def piecewise(df, breaks):
    """
    Performs piecewise regression oh the data and picks the slope over the largest time period.

    :param breaks: Number of line segments to fit a line.
    :param df: Data to perform the segmented regression on.
    :return: Plot and the slope of the desired region.
    """
    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.scatter(df['Times'], df['Width'], label='Original Data', color="blue")

    # Smooth the data
    df = linspaceSmoother(df)

    # Slice the DataFrame up to the truncate_index
    truncate_index = int(len(df) * 0.90)
    df = df.iloc[:truncate_index]

    # Initialize PiecewiseLinFit
    myPWLF = pwlf.PiecewiseLinFit(df['Times'], df['Width'], seed=1975)

    # Fit the data for n line segments
    myPWLF.fit(breaks)

    # Calculate slopes
    slopes = myPWLF.calc_slopes()

    # Find the largest time period
    largest_segment_idx = np.argmax(np.diff(myPWLF.fit_breaks))

    # Get the slope of the largest time period
    largest_segment_slope = slopes[largest_segment_idx]

    # Predict for the determined points
    xHat = df['Times']
    yHat = myPWLF.predict(xHat)

    # Plot smoothed data
    plt.scatter(df['Times'], df['Width'], label='Smoothed Data', color='black')

    # Plot regression line
    plt.plot(xHat, yHat, color='yellow', label='Piecewise Linear Fit')

    # Color the largest segment in green
    largest_segment_start = myPWLF.fit_breaks[largest_segment_idx]
    largest_segment_end = myPWLF.fit_breaks[largest_segment_idx + 1]
    plt.fill_between(
        [largest_segment_start, largest_segment_end],
        df['Width'].min(), df['Width'].max(),  # Fill the entire column
        color='green', alpha=0.2, label='Desired Segment'
    )

    plt.xlabel('Times (ms)')
    plt.ylabel('Width (mm)')
    plt.title('Piecewise Linear Regression')
    plt.legend()
    plt.grid()
    # plt.show()

    return plt, largest_segment_slope, slopes


def linspaceSmoother(df):
    """
    Smooths data frame using linear interpolation of stepped time series data.

    :param df: Data to be smoothed.
    :return: The smoothed dataframe.
    """
    # Create a copy of the DataFrame 'df' to 'processed_df'
    processed_df = df.copy()

    # Calculate the differences between consecutive 'Width' values and create a new 'Diff' column
    df['Diff'] = df['Width'].diff()

    # Group the data by 'Width' and calculate the median of 'Times' for each group
    # Convert the median values to a list and assign them to 'step_centers'
    step_centers = df.groupby('Width')['Times'].median().tolist()

    # ==============================================================================================================
    # plt.scatter(df['Times'], df['Width'], label='Original Data', s=5, color="red")
    # plt.vlines(step_centers, ymin=df['Width'].min(), ymax=df['Width'].max(), color='grey', label='Step Centers')
    # ==============================================================================================================

    # Initialize an empty list to store step widths
    step_widths = []

    # Reverse the order of step_centers list
    step_centers = step_centers[::-1]

    # Iterate through each center value
    for center in step_centers:
        # Find the width corresponding to the center time
        width = df[df['Times'] == center]['Width'].values

        if len(width) == 0:
            # If no width value corresponds to the center time, find the nearest width
            nearest_width = df.iloc[(df['Times'] - center).abs().argsort()[:1]]['Width'].values

            if len(nearest_width) > 0:
                # If the nearest width is found, append it to the step_widths list
                step_widths.append(nearest_width[0])
        else:
            # If a width value corresponds to the center time, append it to the step_widths list
            step_widths.append(width[0])

    # Iterate through the step_widths list to interpolate between each pair of adjacent step centers
    for i in range(len(step_widths) - 1):
        # Select the rows in the original DataFrame (df) that are within the current step range
        df_for_x = df[(df['Times'] >= step_centers[i]) & (df['Times'] <= step_centers[i + 1])]
        count_x = len(df_for_x)

        # Create arrays of x and y values for interpolation
        x_vals = np.linspace(step_centers[i], step_centers[i + 1], num=count_x)
        y_vals = np.linspace(step_widths[i], step_widths[i + 1], num=count_x)

        # Update the 'Times' and 'Width' columns in the processed_df using the interpolated values
        processed_df.loc[df_for_x.index, 'Times'] = x_vals
        processed_df.loc[df_for_x.index, 'Width'] = y_vals

    # ==============================================================================================================
    #     plt.plot(x_vals, y_vals, color='green')
    #
    # plt.scatter(processed_df['Times'], processed_df['Width'], label='Interpolation Data', s=10, color="blue")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # ==================================================================================================================

    # Select only the 'Times' and 'Width' columns from the processed_df DataFrame
    processed_df = processed_df[["Times", "Width"]]

    # Redundant line: Assign processed_df to itself, which doesn't change anything
    processed_df = processed_df

    # Return the processed DataFrame
    return processed_df


def standardize(df, x_col, y_col):
    """
    Standardizes the data since we are using Euclidean distances to flag bad data.

    :param df: Dataframe to be standardized
    :param x_col: X column
    :param y_col: Y column
    :return: Scaled data.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Apply standardization to the specified columns
    df_copy[[x_col, y_col]] = scaler.fit_transform(df_copy[[x_col, y_col]])

    # Return the standardized DataFrame
    return df_copy


def sumDistances(df, plot=False):
    """
    Finds the total sum of the distances between the points to flag data.

    :param df: Dataframe to measure distances in.
    :param plot: True if you want to see the plot.
    :return: True iff the data is flagged.
    """
    # Truncate the DataFrame to the last 20% of rows
    truncate_index = int(len(df) * 0.80)
    df = df.iloc[truncate_index:]

    # Standardize the 'Times' and 'Width' columns
    df = standardize(df, 'Times', 'Width')

    # Extract the feature values ('Times' and 'Width') as a numpy array
    X = df[['Times', 'Width']].values

    # Calculate the pairwise Euclidean distances between data points
    distances = distance.cdist(X, X, 'euclidean')

    # Calculate the sum of distances
    sum_of_distances = distances.sum()

    # Calculate the scaled threshold by dividing the sum of distances by the number of data points
    num_points = len(X)
    scaled_threshold = sum_of_distances / num_points

    # Set a flag to True if the scaled threshold is less than 100, otherwise, set it to False
    flag = False
    if scaled_threshold < 100:
        flag = True

    # If the 'plot' argument is True, create a scatter plot with pairwise distance lines
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], label='Data Points', color='blue')

        # Plot pairwise distance lines (red dashed lines)
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'r--', alpha=0.01)

        # Set plot title and labels
        plt.title(f'Sum of Distances: {sum_of_distances}, Flag: {flag}')
        plt.xlabel('Times')
        plt.ylabel('Width')
        plt.legend()
        plt.show()

    # Print the scaled threshold and flag
    # print(f"Threshold:", scaled_threshold, flag)

    # Return the flag value
    return flag


def flagFiles(directory_path):
    """
    Flags files in this directory.

    :param directory_path: Path to directory.
    :return: Number of flagged files
    """
    # Initialize a count to keep track of flagged files
    count = 0

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory_path):
        # Check if 'OriginalData.csv' is present in the list of files
        for filename in fnmatch.filter(files, 'OriginalData.csv'):
            # Construct the full file path
            file_path = os.path.join(root, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Call the sumDistances function to check if the file should be flagged
            flag = sumDistances(df)

            # If the file is flagged, increment the count and move it to a new folder
            if flag:
                count += 1

                # Extract the folder name from the file path
                folder_name = os.path.basename(os.path.dirname(file_path))

                # Create a new folder name with a '0_FLAG_' prefix
                new_folder_name = '0_FLAG_' + folder_name

                # Construct the path for the new folder
                new_folder_path = os.path.join(directory_path, new_folder_name)

                # Move the folder to the new location
                shutil.move(root, new_folder_path)

    # Return the count of flagged files
    return count


data = pd.read_csv("/Users/ypm/Desktop/DoS/test_big/0.01wt_ wsr 301, 4 Aug 2023 YM  (1)/OriginalData.csv")
data.dropna(inplace=True)


def break_and_fit(data):
    """
    Piecewise fit works better, but this finds breakpoints automatically.

    :param data: Data to fit.
    :return: Slope and plot.
    """
    # Calculate the number of data points in each segment
    segment_size = len(data) // 10

    # Initialize variables to track sum_section values and marked points
    sum_section = []
    marked_points = []

    for i in range(0, len(data), segment_size):
        segment = data.iloc[i:i + segment_size]
        sum_section.append(segment['Width'].sum())

    # Find the threshold for marking points
    mad = np.median(np.abs(np.diff(sum_section)))
    threshold = 1 * mad

    # Mark points where the difference between consecutive sum_section values is too high
    for i in range(1, len(sum_section)):
        if abs(sum_section[i] - sum_section[i - 1]) > threshold:
            marked_points.append(data.iloc[i * segment_size]['Times'])

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(data['Times'], data['Width'], label='Data')

    # Mark the points on the plot
    intervals = []
    if marked_points:
        intervals.append((data['Times'].iloc[0], marked_points[0]))
        for i in range(len(marked_points) - 1):
            intervals.append((marked_points[i], marked_points[i + 1]))
        intervals.append((marked_points[-1], data['Times'].iloc[-1]))

    # Step 2: Find the longest interval without marked points
    longest_interval = max(intervals, key=lambda interval: interval[1] - interval[0])

    # Extract the data within the longest interval
    start_time, end_time = longest_interval
    filtered_data = data[(data['Times'] >= start_time) & (data['Times'] <= end_time)]

    # Fit a linear line to the data within the longest interval
    coefficients = np.polyfit(filtered_data['Times'], filtered_data['Width'], 1)
    linear_fit = np.poly1d(coefficients)

    # Step 3: Plot the data and the linear fit
    plt.figure(figsize=(12, 6))
    plt.plot(data['Times'], data['Width'], label='Data')
    plt.scatter(marked_points, [data[data['Times'] == t]['Width'].values[0] for t in marked_points], color='red',
                label='Marked Points')
    plt.plot(filtered_data['Times'], linear_fit(filtered_data['Times']), label='Linear Fit', color='green',
             linewidth=2.0)

    plt.xlabel('Times')
    plt.ylabel('Width')
    plt.legend()

    slope = coefficients[0]
    return plt, slope, []
