import cv2
import numpy as np


if __name__ == '__main__':
    video_file = '/home/ssilvari/Downloads/test.webm'
    cap = cv2.VideoCapture(video_file)

    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'Video file {video_file}')
    print(f'number of frames {n_frames}')
    print(f'Frame rate {fps}')

    # Create background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
    bg_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Extract background
            fgmask = fgbg.apply(frame)
            bg_frames.append(fgmask)

            cv2.imshow('Original Frame', frame)
            cv2.imshow('Background', fgmask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            mean_sil = np.mean(bg_frames, axis=0)
            bg_frames = []
            cv2.imwrite('/tmp/background_subtracted.png', mean_sil)

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == n_frames:
            break
    cap.release()
    cv2.destroyAllWindows()
