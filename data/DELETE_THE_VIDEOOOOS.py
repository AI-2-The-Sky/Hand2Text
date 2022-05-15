import os

content = os.listdir('H2T/raw_videos')

print("CAREFULL IT'S DELETING YOUR STUFF :O")

for i, file in enumerate(content, 1):
  if (i % 2 == 0):
    os.remove(f'H2T/raw_videos/{file}')