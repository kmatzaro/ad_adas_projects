from moviepy import VideoFileClip

clip = VideoFileClip("E:\\ad_adas_projects\\perception_ad\\recordings\\lane_detection_20250805_111433.mp4") .subclipped(0, 90)  # optional: crop first 10 sec
clip = clip.resized(width=480)  # scale down for GIF size
clip.write_gif(".\\perception_ad\\demo\\perception_demo.gif", fps=10)
