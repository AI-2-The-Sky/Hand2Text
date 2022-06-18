import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
VID_DIR = f"{ROOT_DIR}/data/H2T/raw_videos"

content = os.listdir(VID_DIR)

print("CAREFULL IT'S DELETING YOUR STUFF :O")

for i, file in enumerate(content, 1):
    if i % 2 == 0:
        os.remove(f"{VID_DIR}/{file}")
