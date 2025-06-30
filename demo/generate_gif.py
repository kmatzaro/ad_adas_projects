from moviepy.editor import VideoFileClip

clip = VideoFileClip("E:\\ad_adas_projects\\perception_autonomous_driving\\lane_detection_20250630_143354.mp4").subclip(0, 10)  # optional: crop first 10 sec
clip = clip.resize(width=480)  # scale down for GIF size
clip.write_gif("./lane_detection_demo.gif", fps=10)
