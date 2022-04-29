import os

import cv2

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def get_frame_from_video(filepath: str, frame_subdir: str):
    vid = cv2.VideoCapture(filepath)
    current_frame = 0

    while True:
        success, frame = vid.read()
        if not success:
            break
        elif current_frame % 25 == 0:
            full_filepath = f"{frame_subdir}/frame-{current_frame}.jpg"
            cv2.imwrite(full_filepath, frame)
        current_frame += 1

    vid.release()
    cv2.destroyAllWindows()
    return


def main():
    SUB_DIR = f"{ROOT_DIR}/data/H2T"
    FRAMES_DIR = f"{SUB_DIR}/frames"
    RAW_VIDEOS_PATH = f"{SUB_DIR}/raw_videos"
    all_file = os.listdir(RAW_VIDEOS_PATH)
    len_all_file = len(all_file)

    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
    for i, file in enumerate(all_file, 1):
        frame_subdir = f"{FRAMES_DIR}/{file.split('.')[0]}"
        print("Extract frames from %s: %.2f%%" % (file, i * 100 / len_all_file))
        if os.path.exists(frame_subdir):
            continue
        else:
            os.makedirs(frame_subdir)
        get_frame_from_video(f"{RAW_VIDEOS_PATH}/{file}", frame_subdir)


if __name__ == "__main__":
    main()
