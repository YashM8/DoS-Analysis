import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def linspaceSmoother(df):
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

    # Initialize an empty list to store interpolated step widths
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
        # plt.plot(x_vals, y_vals, color='green')

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


def find_appropriate_slope(lst):
    """
    Finds the most similar numbers in the list (within 80%).

    :param lst: List of slopes.
    :return: Mean of the most similar slope.
    """
    print(lst)
    slope = []  # Initialize an empty list to store pairs of values that meet the ratio condition

    # Loop through the list up to the second-to-last element
    for i in range(len(lst) - 1):
        ratio = lst[i] / lst[i+1]  # Calculate the ratio of the current element to the next element
        # print(f"{lst[i]}/{lst[i+1]} = {ratio}")

        # Check if the ratio is within the range [0.8, 1.2]
        if 1.3 >= ratio >= 0.7:
            slope.append(lst[i])  # Append the current element to the slope list
            slope.append(lst[i+1])  # Append the next element to the slope list

    # Calculate the average of the values that met the ratio condition
    return sum(slope) / len(slope)


def autoRegressor(orig_data):
    """
    Finds the slope of the linear part of the 2D data automatically.

    :param orig_data: Dataframe to detect linear part from.
    :return: The slope of the linear part of the 2D data.
    """
    # Smoothed data is prepared using the linspaceSmoother function and assigned to smoothed_data DataFrame
    smoothed_data = linspaceSmoother(orig_data)

    # Extract 'Times' and 'Width' columns from the smoothed_data DataFrame
    x = smoothed_data["Times"]
    y = smoothed_data["Width"]

    # Initialize empty lists to store slopes and intercepts of linear regressions
    slopes = []
    intercepts = []

    # Define the window size for the linear regression analysis
    window_size = len(x) // 10

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

    # Remove 0's from the list of slopes.
    slopes = list(filter(lambda x: x != 0, slopes))

    # Return the plot object and the result of the find_appropriate_slope function
    return plt, find_appropriate_slope(slopes)
