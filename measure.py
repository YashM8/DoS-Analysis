import cv2
from statistics import mean, median
import pandas as pd
import numpy as np
import utils_find_1st as utf1st
import matplotlib.pyplot as plt


def almostSame(num1, num2, num3):
    """
    Returns True only if the three numbers are within 5% of each other.

    :param num1: Number one
    :param num2: Number two
    :param num3: number three
    :return:
    """
    # Check if all three input numbers are equal.
    if num1 == num2 == num3:
        # If they are equal, return True.
        return True

    # Find the maximum value among the three input numbers.
    max_value = max(num1, num2, num3)

    # Find the minimum value among the three input numbers.
    min_value = min(num1, num2, num3)

    # Calculate a range threshold as 5% of the difference between the maximum and minimum values.
    range_threshold = (max_value - min_value) * 0.05

    # Check if the absolute difference between num1 and num2 is within the range threshold/
    return (abs(num1 - num2) <= range_threshold and
            abs(num1 - num3) <= range_threshold and
            abs(num2 - num3) <= range_threshold)


def measureWidths(filename, needle_mm, fps, show=False, skip=1):
    """
    Measures the width of the smallest part of the droplet.

    :param fps: Original FPS f the video.
    :param needle_mm: Needle width in millimeter.
    :param filename: File to process.
    :param show: Show the video with the boundary overlay. Default is False.
    :param skip: Number of columns to skip. Default is None.
    :return: A dataframe with Times and Widths.
    """
    # Create a VideoCapture object to open and read the specified video file
    cap = cv2.VideoCapture(filename)

    # Initialize empty lists to store frames and their corresponding widths
    frames, widths = [], []

    # Define the frames per second and needle width in millimeters
    frames_per_second = fps
    needle_mm = needle_mm

    np.seterr(all='ignore')

    # Loop to read frames from the video file
    while cap.isOpened():
        # Read the next frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break  # Break the loop if there are no more frames to read

        # Get the dimensions of the frame
        rows, cols = frame.shape[:2]

        # Convert the color frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ========================================================
        t_vals = []
        coordinates = [i * 30 + 10 for i in range(8)]
        for x in coordinates:
            t_vals.append(gray_frame[25, x])
            frame[25, x] = [0, 255, 0]
            t_vals.append(gray_frame[rows - 25, x])
            frame[rows - 25, x] = [0, 255, 0]
        t_val = int(mean(t_vals))
        threshold = t_val - len(t_vals) - 2
        # ========================================================

        # Apply binary thresholding using the Triangle method and create a binary mask
        _, binary = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY_INV)  # + cv2.THRESH_TRIANGLE
        binary_frame = binary > 0

        # Initialize an empty list to store widths of features in the frame
        widths_in_frame = []

        # Loop through columns of the frame, skipping by 'skip'
        for column in range(5, cols - 5, skip):
            # Find the first non-zero pixel in the column (top to bottom)
            center_row_bottom = utf1st.find_1st(binary_frame[:, column],  # List to look for first element.
                                                1,  # What to look for.
                                                utf1st.cmp_equal)  # Check if equal.
            if center_row_bottom != -1:
                frame[center_row_bottom, column] = [0, 0, 255]

            # Find the first non-zero pixel in the column (bottom to top)
            center_row_top = utf1st.find_1st(binary_frame[:, column][::-1], 1, utf1st.cmp_equal)
            if center_row_top != -1:
                frame[rows - center_row_top - 1, column] = [0, 0, 255]

            # Find the first non-zero pixel in the left column (top to bottom)
            left_row_bottom = utf1st.find_1st(binary_frame[:, column - 1], 1, utf1st.cmp_equal)
            left_row_top = utf1st.find_1st(binary_frame[:, column - 1][::-1], 1, utf1st.cmp_equal)

            # Find the first non-zero pixel in the right column (top to bottom)
            right_row_bottom = utf1st.find_1st(binary_frame[:, column + 1], 1, utf1st.cmp_equal)
            right_row_top = utf1st.find_1st(binary_frame[:, column + 1][::-1], 1, utf1st.cmp_equal)

            # Calculate each row's width to verify the measurement
            width = rows - center_row_top - 1 - center_row_bottom
            left_width = rows - left_row_top - 1 - left_row_bottom
            right_width = rows - right_row_top - 1 - right_row_bottom

            if almostSame(width, left_width, right_width):
                widths_in_frame.append(width)

            # Calculate the width of the feature in this column and add it to the list
            widths_in_frame.append(rows - center_row_top - 1 - center_row_bottom)

        # Calculate the minimum width in the frame and convert it to logarithmic scale
        min_width = min(widths_in_frame)
        log_val = np.log(min_width / needle_mm)

        # Append the calculated log value and the frame timestamp to the respective lists
        widths.append(log_val)
        frames.append((cap.get(cv2.CAP_PROP_POS_FRAMES) / frames_per_second) * 1000)

        if show:
            # Set a window title based on the width of the frame
            cv2.imshow(f"Overlay Frame - {cols} Pixels Wide", frame)

            # Wait for a key press and check if the pressed key is 'q' (ASCII value 113)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if show:
        plt.scatter(frames, widths, s=6, marker='o', label='Data Points', color="red")
        plt.show()

    # Create a DataFrame using the 'widths' and 'frames' lists
    df = pd.DataFrame({'Width': widths, 'Times': frames})

    # Replace infinite and negative infinite values with NaN and then drop rows with NaN values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # df.to_csv('analysis_results.csv', index=False)

    if show:
        # If 'show' parameter is True, release the video capture object and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

    # Return the DataFrame
    return df


measureWidths("/Users/ypm/Desktop/test files/problem.mp4", needle_mm=2.11, fps=2999,
              show=False, skip=1)
