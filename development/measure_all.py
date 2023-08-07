"""
For testing.

Final version in 'analyze.py'
"""
import os
import cv2
import pandas as pd
import numpy as np
from statistics import mean
import utils_find_1st as utf1st
import matplotlib.pyplot as plt


def are_within_5_percent(a, b, c):
    if a == b == c:
        return True
    max_num = max(a, b, c)
    min_num = min(a, b, c)

    percentage_diff = ((max_num - min_num) / max_num) * 100
    return percentage_diff <= 5


def measureWidths(filename, needle_mm, fps, show=False, skip=1):
    cap = cv2.VideoCapture(filename)
    frames, widths = [], []
    frames_per_second = fps
    needle_mm = needle_mm

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rows, cols = frame.shape[:2]

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        binary_frame = binary > 0

        widths_in_frame = []

        for c in range(5, cols - 5, skip):

            r = utf1st.find_1st(binary_frame[:, c], 1, utf1st.cmp_equal)
            if r != -1: frame[r, c] = [0, 0, 255]

            ro = utf1st.find_1st(binary_frame[:, c][::-1], 1, utf1st.cmp_equal)
            if ro != -1: frame[rows - ro - 1, c] = [0, 0, 255]

            width = rows - ro - 1 - r
            widths_in_frame.append(width)

        min_width = min(widths_in_frame) * 0.00703333333
        log_val = np.log(min_width / needle_mm)
        widths.append(log_val)
        frames.append((cap.get(cv2.CAP_PROP_POS_FRAMES) / frames_per_second) * 1000)

        if show:
            cv2.imshow(f"Overlay Frame - {cols} Pixels Wide", binary)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Exit by key')
                break

    plt.scatter(frames, widths, s=6, marker='o', label='Data Points', color="red")
    plt.show()

    df = pd.DataFrame({'Width': widths, 'Times': frames})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(len(widths))
    if show:
        cap.release()
        cv2.destroyAllWindows()
    return df


video_file_path = "/Users/ypm/Desktop/try.mp4"
needle_width = 1.9
fps = 2999
measureWidths(video_file_path, needle_mm=needle_width, fps=fps, show=False, skip=1)

# column = binary_frame[:, c]
# column_left = binary_frame[:, c - 1] if c > 0 else None
# column_right = binary_frame[:, c + 1] if c < cols - 1 else None

# r = utf1st.find_1st(column_left, 1, utf1st.cmp_equal)
# ro = utf1st.find_1st(column[::-1], 1, utf1st.cmp_equal)


# getThreshold([146, 110, 102])

# t_vals = []
# x_coords = [10, 60, 110, 160, 210, 260, 310, 360]
# for x in x_coords:
#     t_vals.append(gray_frame[25, x])
# t_val = int(mean(t_vals))
# threshold = t_val - (t_val//15)
