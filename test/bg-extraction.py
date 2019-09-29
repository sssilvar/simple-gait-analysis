import cv2
import numpy as np
import pandas as pd
from scipy import signal
from pdlab.peakdetect import peakdet

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

if __name__ == '__main__':
    video_file = '/home/ssilvari/Downloads/carlos1.mp4'
    cap = cv2.VideoCapture(video_file)

    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'Video file {video_file}')
    print(f'number of frames {n_frames}')
    print(f'Frame rate {fps}')

    # Create background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
    bg_frames = []

    # Series with data
    features = pd.DataFrame()

    print('Processing video...')
    while cap.isOpened():
        ret, frame = cap.read()

        # Get current Frame
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time_min = cap.get(cv2.CAP_PROP_POS_MSEC) / 6e4  # ms -> min

        if ret:
            # Smooth it out
            frame = cv2.GaussianBlur(frame, (15, 15), cv2.BORDER_DEFAULT)
            # Extract background
            fgmask = fgbg.apply(frame)
            top, bottom = np.split(fgmask, 2, axis=0)
            # print(top.shape, bottom.shape, fgmask.shape)

            cv2.imshow('Original Frame', frame)
            # cv2.imshow('Background', fgmask)
            cv2.imshow('Bottom', bottom)

            # Add features
            if current_frame > 300:
                features.loc[current_time_min, 'N_f'] = (bottom != 0).sum()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('/tmp/feet.png', bottom)

        if current_frame == n_frames:
            print()
            break
        elif current_frame % 200 == 0:
            print(f'{current_frame:.0f}..', end='')
    cap.release()
    cv2.destroyAllWindows()

    # Moving average to smooth out the signal
    series = features['N_f']
    series_mov_avg = series.rolling(window=int(fps * 2)).mean()

    # Convolution filter
    window = signal.gaussian(40, std=9)
    filtered = signal.convolve(series, window, mode='same') / sum(window)

    series.plot(label='Gait kinematics', alpha=0.2)
    plt.plot(series.index, filtered, label='Gaussian Filtered', linewidth=2)

    # Find minimum and maximum points (peaks and valleys)
    maxtab, mintab = peakdet(filtered, len(series))
    # print(mintab)

    valley_x = series.iloc[mintab[:, 0]].index
    valley_y = mintab[:, 1]
    plt.scatter(valley_x, valley_y, color='green', label='Single Stance', s=50)

    cadence = len(valley_x) / np.mean(np.diff(valley_x))
    print(f'Mean cadence: {cadence:.2f} steps/min')

    plt.xlabel('Time [minutes]')
    plt.ylabel('Number of background pixels')
    plt.legend()

    plt.show()
