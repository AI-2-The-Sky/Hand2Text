import os

import cv2


def get_frame_from_video(filepath, name_file):
    vid = cv2.VideoCapture(filepath)
    current_frame = 0
    name_dir = name_file[0:5]

    while True:
        success, frame = vid.read()
        if success and current_frame % 25 == 0 and current_frame != 0:
            full_filepath = "./frame/" + name_dir + "/frame" + str(current_frame) + ".jpg"
            cv2.imwrite("./frame/" + name_dir + "/frame" + str(current_frame) + ".jpg", frame)
        current_frame += 1
        if not success:
            break

    vid.release()
    cv2.destroyAllWindows()
    return


compt = 0
all_file = os.listdir("./data/raw_videos/")
len_all_file = len(all_file)
if not os.path.exists("./frame/"):
    os.makedir("./frame")
for file in all_file:
    if not os.path.exists("./frame/" + file[0:5]):
        os.makedirs("./frame/" + file[0:5])
    print("Extract frames from", file, "{:.2f}".format(compt * 100 / len_all_file), "%")
    get_frame_from_video("./data/raw_videos/" + file, file)
    compt += 1
