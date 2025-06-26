import sys, glob, os
import numpy as np
import cv2
import time

try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def sliding_window_polyfit(binary_warped, nwindows=9, margin=100, minpix=50):
    # 1. Histogram to find base points
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 2. Set window height
    window_height = int(binary_warped.shape[0] / nwindows)

    # Identify nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Find nonzero pixels in window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter if enough pixels found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate all indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a 2nd degree polynomial
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None

    print("Left fit:", left_fit)
    print("Right fit:", right_fit)

    return left_fit, right_fit

def get_bev_transform(img_size):
    src = np.float32([
        [img_size[0] * 0.48, img_size[1] * 0.58],
        [img_size[0] * 0.52, img_size[1] * 0.58],
        [img_size[0] * 0.75, img_size[1]],
        [img_size[0] * 0.25, img_size[1]]
    ])

    dst = np.float32([
        [img_size[0] * 0.25, 0],
        [img_size[0] * 0.75, 0],
        [img_size[0] * 0.75, img_size[1]],
        [img_size[0] * 0.25, img_size[1]]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src, dst

def draw_lane_overlay(image, warped_binary, left_fit, right_fit, Minv):
    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    color_warp = np.zeros_like(image).astype(np.uint8)

    # Evaluate polynomial fits
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast into format cv2.fillPoly() wants
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, [pts.astype(np.int32)], (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result

def process_lane_image(image):

    # Convert BGRA CARLA image to BGR
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame = array[:, :, :3]

    # Resize (optional, for performance)
    frame = cv2.resize(frame, (1080, 720))

    # Get BEV transform matrix
    img_size = (frame.shape[1], frame.shape[0])
    M, Minv, src, dst = get_bev_transform(img_size)

    # Warp to BEV
    warped = cv2.warpPerspective(frame, M, img_size)
    cv2.imshow("Warped BEV", warped)
    cv2.waitKey(1)

    # Continue with preprocessing on warped
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Region of interest
    mask = np.zeros_like(edges) 
    height, width = edges.shape
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)
    cv2.imshow("masked", masked)

    # Fit lanes with sliding window
    left_fit, right_fit = sliding_window_polyfit(masked)

    # Draw overlay and warp it back
    result = draw_lane_overlay(frame, warped, left_fit, right_fit, Minv)
    cv2.imshow("Lane Overlay", result)
    cv2.waitKey(1)

actors = []

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town03')
    blueprint_library = world.get_blueprint_library()

    # Find an RGB camera blueprint
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '105')

    # Find a vehicle and attach camera
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True)
    actors.append(vehicle)

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actors.append(camera)

    camera.listen(lambda image: process_lane_image(image))

    print("Press Ctrl+C to exit...")
    try:
        while True:
            world.tick()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        print("Cleaning up...")
        for actor in actors:
            if actor.is_alive:
                actor.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()