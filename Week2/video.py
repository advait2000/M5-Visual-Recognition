import cv2
import os

output_file = 'outputfinetune.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 3
frame_size = (1490, 450)  # replace with actual dimensions of frames
video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

frame_dir = '/Users/advaitdixit/Documents/Masters/M5-Visual-Recognition/Week2/op1-finetuning'
count = 0
for filename in os.listdir(frame_dir):
    count += 1
    if count < 100:
        img = cv2.imread(os.path.join(frame_dir, filename))
        print(img.shape)
        img = cv2.resize(img, (1490, 450))
        video_writer.write(img)
