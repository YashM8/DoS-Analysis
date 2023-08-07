import cv2
import pandas as pd
import numpy as np
import utils_find_1st as utf1st


def almostSame(num1, num2, num3):
    if num1 == num2 == num3:
        return True
    max_value = max(num1, num2, num3)
    min_value = min(num1, num2, num3)
    range_threshold = (max_value - min_value) * 0.05

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

    # Loop to read frames from the video file
    while cap.isOpened():
        # Read the next frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret: break  # Break the loop if there are no more frames to read

        # Convert the color frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ========================================================
        # t_vals = []
        # x_coords = [10, 60, 110, 160, 210, 260, 310, 360]
        # for x in x_coords:
        #     t_vals.append(gray_frame[25, x])
        # t_val = int(mean(t_vals))
        # threshold = t_val - (t_val//15)
        # ========================================================

        # Apply binary thresholding using the Triangle method and create a binary mask
        _, binary = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        binary_frame = binary > 0

        # Get the dimensions of the frame
        rows, cols = frame.shape[:2]

        # Initialize an empty list to store widths of features in the frame
        widths_in_frame = []

        # Loop through columns of the frame, skipping by 'skip'
        for c in range(5, cols - 5, skip):
            # Find the first non-zero pixel in the column (top to bottom)
            r = utf1st.find_1st(binary_frame[:, c], 1, utf1st.cmp_equal)
            if r != -1:
                frame[r, c] = [0, 0, 255]

            # Find the first non-zero pixel in the column (bottom to top)
            ro = utf1st.find_1st(binary_frame[:, c][::-1], 1, utf1st.cmp_equal)
            if ro != -1:
                frame[rows - ro - 1, c] = [0, 0, 255]

            rl = utf1st.find_1st(binary_frame[:, c - 1], 1, utf1st.cmp_equal)
            rol = utf1st.find_1st(binary_frame[:, c - 1][::-1], 1, utf1st.cmp_equal)

            rr = utf1st.find_1st(binary_frame[:, c + 1], 1, utf1st.cmp_equal)
            ror = utf1st.find_1st(binary_frame[:, c + 1][::-1], 1, utf1st.cmp_equal)

            width = rows - ro - 1 - r
            left_width = rows - rol - 1 - rl
            right_width = rows - ror - 1 - rr

            if almostSame(width, left_width, right_width):
                widths_in_frame.append(width)

            # Calculate the width of the feature in this column and add it to the list
            widths_in_frame.append(rows - ro - 1 - r)

        # Calculate the minimum width in the frame and convert it to logarithmic scale
        min_width = min(widths_in_frame) * 0.00703333333
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
    # =================================================================================
    # plt.scatter(frames, widths, s=6, marker='o', label='Data Points', color="red")
    # plt.show()
    # =================================================================================

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
