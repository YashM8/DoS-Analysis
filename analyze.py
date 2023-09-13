import numpy as np
import matplotlib.pyplot as plt
import pwlf


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
