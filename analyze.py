import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from statistics import mean
import pwlf
from scipy import stats


def piecewise(df):
    """
    Performs piecewise regression oh the data and picks the slope over the largest time period.

    NOTE:-
    A stochastic algorithm is used in this function and to ensure reproducible results, please change the 'seed'
    parameter in the 'piecewise.py' library file (in the 'differential_evolution' function within 'fit' function) to
    any random integer.

    ======================================================================================
    piecewise.py (within the pwlf library) >
    fit >
    differential_evolution >
    seed=None to seed=<Any random integer>
    ======================================================================================

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
    myPWLF = pwlf.PiecewiseLinFit(df['Times'], df['Width'])

    # Fit the data for n line segments
    myPWLF.fit(5)

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

    plt.xlabel('Times (s)')
    plt.ylabel('Width (mm)')
    plt.title('Piecewise Linear Regression')
    plt.legend()
    plt.grid()
    # plt.show()

    return plt, largest_segment_slope, slopes


def savGolSmoother(dataframe, polynomial=1):
    """
    Smooths the data using the Savitzky-Golay method.

    :param polynomial: The order of the polynomial used to smooth the data.
    :param dataframe: Data to smooth, inputted as a Panda's dataframe.
    :return: Smoothed data.
    """
    # Extract the 'Times' and 'Width' columns from the input dataframe
    times = dataframe["Times"].values
    width = dataframe["Width"].values

    # Calculate the window length for smoothing
    window_length = len(width) // 8

    # Apply Savitzky-Golay smoothing to the 'Width' column using the specified polynomial order and window length
    smoothed_width = savgol_filter(width, window_length, polynomial)

    # Create a new dataframe with the original 'Times' and the smoothed 'Width' values
    smoothed_df = pd.DataFrame({"Times": times, "Width": smoothed_width})

    # Return the smoothed dataframe
    return smoothed_df


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


def similar(slopes):
    """
    Finds the three most similar number given a list of numbers.

    :param slopes: List of numbers.
    :return: The mean of the 3 most similar numbers.
    """
    # Sort the slopes in ascending order
    sorted_slopes = sorted(slopes)

    # Initialize variables to store the most similar slopes and minimum difference
    most_similar_slopes = []
    min_difference = float('inf')

    # Loop through the sorted slopes, except for the last two
    for i in range(len(sorted_slopes) - 2):
        # Calculate the absolute differences between adjacent slopes
        diff1 = abs(sorted_slopes[i] - sorted_slopes[i + 1])
        diff2 = abs(sorted_slopes[i + 1] - sorted_slopes[i + 2])

        # Calculate the total difference between the adjacent slopes
        total_difference = diff1 + diff2

        # Update the most similar slopes if the total difference is smaller
        if total_difference < min_difference:
            min_difference = total_difference
            most_similar_slopes = [sorted_slopes[i], sorted_slopes[i + 1], sorted_slopes[i + 2]]

    # Calculate and return the mean of the most similar slopes
    return mean(most_similar_slopes)


def ratio(slopes):
    """
     Finds the numbers within 80% of each other given a list of numbers.

    :param slopes: List of numbers.
    :return: The mean of the most similar numbers.
    """
    # Filter out slopes that are less than 0
    slopes = list(filter(lambda x: x < 0, slopes))
    filtered_slopes = []

    # Loop through the filtered slopes, except for the last one
    for i in range(len(slopes) - 1):
        # Calculate the ratio between adjacent slopes
        ratio = slopes[i] / slopes[i + 1]

        # Check if the ratio is within the range of 0.8 to 1.2
        if 1.2 >= ratio >= 0.8:
            filtered_slopes.append(slopes[i])

    # Calculate and return the mean of the filtered slopes
    return mean(slopes)  # Comment indicating the purpose of this code


def autoRegressor(orig_data):
    """
    Finds the slope of the linear part of the 2D data automatically.

    :param orig_data: Dataframe to detect linear part from.
    :return: The slope of the linear part of the 2D data.
    """
    # Smoothed data is prepared using the smoother function and assigned to smoothed_data DataFrame
    # smoothed_data = savGolSmoother(orig_data)
    smoothed_data = linspaceSmoother(orig_data)

    # Extract 'Times' and 'Width' columns from the smoothed_data DataFrame
    x = smoothed_data["Times"]
    y = smoothed_data["Width"]

    # Initialize empty lists to store slopes and intercepts of linear regressions
    slopes = []
    intercepts = []

    # Define the window size for the linear regression analysis
    window_size = len(x) // 7

    # Iterate through sections of the data using the defined window size
    for section_start in range(0, len(x), window_size):
        end = section_start + window_size
        group_x = x[section_start:end].values.reshape(-1, 1)
        group_y = y[section_start:end]

        # Create a LinearRegression model and fit it to the current data section
        lr = LinearRegression()
        lr.fit(group_x, group_y)
        y_pred = lr.predict(group_x)

        # Store the slope and intercept of the linear regression model
        slopes.append(lr.coef_[0])
        intercepts.append(lr.intercept_)

        # Plot the linear regression line and a vertical line at the section's starting point
        plt.plot(group_x, y_pred, color='yellow', linewidth=2)
        plt.axvline(x=group_x[0], color='grey', linestyle='--')

    # Scatter plot the original and smoothed data points
    plt.scatter(orig_data["Times"], orig_data["Width"], label="Original Data", color="red")
    plt.scatter(x, y, label="Smoothed Data", color="blue")

    # Add labels, legend, and title to the plot
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Log(Min Width / Needle Width)")
    plt.title("DoS Radius Evolution")

    # Return the plot object and the result of the find_appropriate_slope function
    orig_slopes = slopes

    # Calculate the average of the values that met the ratio condition
    return plt, ratio(orig_slopes), orig_slopes


def findRegion(dataframe, partitions, threshold):
    """
    Finds the most stable region in 2-D data by calculating derivatives and measuring regions with the least
    fluctuation.

    :param dataframe: Data to find the region in.
    :param partitions: Number of segments to measure fluctuations.
    :param threshold: The threshold below which the fluctuation needs to be below to be a valid segment. The higher the
                    value the more functions are tolerated. This threshold is scaled and I find that values between
                    5 and 10 work best.
    :return: The start and stop time along with the original data to work on.
    """
    orig_data = dataframe  # Store the original data for later use
    dataframe = savGolSmoother(dataframe, 2)  # Apply a Savitzky-Golay smoothing to the data
    dataframe['derivative'] = (dataframe['Width'].diff() / dataframe['Times'].diff())  # Compute the derivative
    dataframe.dropna(inplace=True)  # Remove rows with missing values after differentiation
    differences, x, y = {}, dataframe["Times"], dataframe["derivative"]
    window_size = len(y) // partitions  # Calculate the size of each partition

    # Visualize the data and marked regions for each partition
    plt.figure(figsize=(15, 6))
    plt.plot(x, y, label="Data")
    plt.locator_params(axis='x', nbins=30)

    for section_num, section_start in enumerate(range(0, len(x), window_size)):
        # Extract data for the current partition
        end = section_start + window_size
        group_x, group_y = x[section_start:end].values.reshape(-1, 1), y[section_start:end]

        # Find maximum and minimum points within the partition
        max_y, max_index, max_x = max(group_y), np.argmax(group_y), group_x[np.argmax(group_y)]
        min_y, min_index, min_x = min(group_y), np.argmin(group_y), group_x[np.argmin(group_y)]
        differences[(1000 * (max_y - min_y))] = group_x[0][0]

        # Visualize the maximum and minimum points and annotate the section number
        plt.axvline(x=group_x[0], color='grey', linestyle='--')
        plt.scatter(max_x, max_y, color='r', marker='o', s=30)
        plt.scatter(min_x, min_y, color='r', marker='o', s=30)
        plt.annotate(f"{section_num + 1}", ((group_x[0] + group_x[-1]) / 2, max_y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    # Analyze linear regions based on specified threshold
    linear = []
    prev_value = None

    for value, time in differences.items():
        if float(value) < threshold and (prev_value is None or float(prev_value) < threshold):
            linear.append((value, time))
        prev_value = value

    start = linear[0][1]
    end = linear[-1][1]

    return start, end, orig_data  # Return the start and end times of the linear region along with the original data


def findSlope(df, start, end):
    """
    Finds the slope of the desired region and plots it.

    :param df: The data to perform linear regression on.
    :param start: Start time.
    :param end: Stop time.
    :return: Slope and the plot.
    """
    subset_df = df[(df['Times'] >= start) & (df['Times'] <= end)]  # Extract a subset of data

    X = subset_df[['Times']]
    y = subset_df['Width']

    # Perform linear regression on the subset
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    predicted_width = model.predict(X)

    # Visualize the actual data and linear regression line
    plt.figure(figsize=(15, 6))
    plt.scatter(df['Times'], df['Width'], label='Actual Data')
    plt.plot(subset_df['Times'], predicted_width, color='red', label='Linear Regression Line')
    plt.xlabel('Times')
    plt.ylabel('Width')
    plt.title('Linear Regression and Data Plot')
    plt.legend()
    plt.grid()

    return slope, plt  # Return the calculated slope and the plot object
