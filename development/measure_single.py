import cv2
import time
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def measureWidth(video_file, threshold_value, column_position):
    cap = cv2.VideoCapture(video_file)
    widths, frames_with_overlay, times_ms = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

        white_top, white_bottom = 0, 0
        height, column = binary_frame.shape

        for row in range(0, height):
            if binary_frame[row, column_position] == 255:
                white_bottom += 1
                frame[row, column_position] = [0, 255, 0]
                if binary_frame[row + 1, column_position] == 0: break

        for row in reversed(range(0, height)):
            if binary_frame[row, column_position] == 255:
                white_top += 1
                frame[row, column_position] = [0, 255, 0]
                if binary_frame[row - 1, column_position] == 0: break

        width = (height * 1) - (white_top + white_bottom)
        widths.append(width)

        cv2.imshow("Frame", frame)

        times_ms.append((cap.get(cv2.CAP_PROP_POS_FRAMES)))

    cap.release()
    data = {'Width': widths, 'Time': times_ms}
    df = pd.DataFrame(data)
    return df


def visualizeWidths(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time'], df['Width'], 'bo', markersize=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Measured Width (mm)')
    plt.title('Radius Evolution')
    plt.grid(True)
    plt.show()

    # for frame in frames_with_overlay:
    #     cv2.imshow('Video with Overlay', frame)
    #     # time.sleep(0.001)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27: break
    #
    # cv2.destroyAllWindows()


video_file_path = "/Users/ypm/Desktop/new/testfile.mp4"
threshold = 100
col_num = 550
drop_data = measureWidth(video_file_path, threshold, col_num)
visualizeWidths(drop_data)
# time_taken = timeit.timeit(lambda: measureWidth(video_file_path, threshold, col_num), number=1)
# print(f"Time taken: {time_taken:.2f} seconds")
print("done")


def replace_stepped_data(dataframe, threshold=1e-3):
    df = dataframe.copy()
    diff_values = df['Width'].diff()
    steps = (np.abs(diff_values) > threshold)
    groups = (steps != steps.shift()).cumsum()

    for group_num in range(1, groups.max() + 1):
        group = df[groups == group_num]
        next_group = df[groups == (group_num + 1)]
        num_points = len(group['Times'])

        if len(group) > 10:
            center_time = group['Times'].median()
            next_center_time = next_group["Times"].median()

            start_idx = df.loc[df['Times'] == center_time, 'Width'].first_valid_index()
            if start_idx is not None:
                start = df.loc[start_idx, 'Width']
                plt.axvline(x=center_time, color='gray', linestyle='--')

            stop_idx = df.loc[df['Times'] == next_center_time, 'Width'].first_valid_index()
            if stop_idx is not None:
                stop = df.loc[stop_idx, 'Width']
                plt.axvline(x=center_time, color='gray', linestyle='--')

            spaced_values = np.linspace(start, stop, num_points)
            df.loc[group.index, 'Width'] = spaced_values
            # plt.axvline(x=center_time, color='gray', linestyle='--')

    plt.scatter(dataframe['Times'], dataframe['Width'], label='Original Data', color='red', marker='o')
    plt.plot(df["Times"], df["Width"], label='Linspace Data', color='blue')
    plt.title('Smoothed Data over Original Data')
    plt.legend()
    plt.grid()
    plt.show()

    return df.groupby(groups)
