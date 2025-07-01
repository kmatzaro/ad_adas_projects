from moviepy.editor import VideoFileClip

clip = VideoFileClip("E:\\ad_adas_projects\\perception_autonomous_driving\\lane_detection_20250701_204103.mp4").subclip(30, 90)  # optional: crop first 10 sec
clip = clip.resize(width=480)  # scale down for GIF size
clip.write_gif(".\\perception_autonomous_driving\\demo\\lane_detection_demo_2.gif", fps=10)
