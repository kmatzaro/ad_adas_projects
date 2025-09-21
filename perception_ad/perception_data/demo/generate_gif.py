from moviepy import VideoFileClip

clip = VideoFileClip(r"E:/ad_adas_projects/perception_ad/perception_data/recordings/multi_sensor_20250921_194536.mp4")# .subclipped(0, 90)  # optional: crop first 10 sec
clip = clip.resized(width=480)  # scale down for GIF size
clip.write_gif(r"./perception_data/demo/multi_sensor_demo.gif", fps=10)
