from moviepy.editor import VideoFileClip

clip = VideoFileClip("E:\\ad_adas_projects\\perception_ad\\recordings\\lane_detection_20250720_191858.mp4").subclip(95, 130)  # optional: crop first 10 sec
clip = clip.resize(width=480)  # scale down for GIF size
clip.write_gif(".\\perception_ad\\demo\\lane_detection_demo.gif", fps=10)
