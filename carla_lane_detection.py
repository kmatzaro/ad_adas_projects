import pygame
import numpy as np
import sys, glob, os
import cv2
from threading import Thread
from simple_lane_detection import SimpleLaneDetector
import time 
import random

try:
    sys.path.append(glob.glob('../carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarlaLaneDetection:
    def __init__(self):
        self.actors = []
        self.lane_detector = SimpleLaneDetector((1920, 1080))

    def run(self):
        pygame.init()
        display = pygame.display.set_mode(self.lane_detector.img_size)
        pygame.display.set_caption("CARLA Lane Detection")

        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town03')
        blueprint_library = world.get_blueprint_library()

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '105')

        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
        vehicle.set_autopilot(True)
        self.actors.append(vehicle)

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        self.actors.append(camera)

        def camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame = array[:, :, :3]
            result, gray, edges, masked = self.lane_detector.process_image(frame)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(np.rot90(result))
            display.blit(surface, (0, 0))

            # Create debug miniatures
            def draw_debug(title, img, x_offset):
                debug_img = cv2.resize(img, (160, 120))
                debug_surface = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)))
                display.blit(debug_surface, (self.lane_detector.img_size[0]-200, x_offset))
                text = pygame.font.SysFont(pygame.font.get_default_font(), 16).render(title, True, (255, 255, 255))
                display.blit(text, (self.lane_detector.img_size[0]-200, x_offset + 120))

            draw_debug("Gray", gray, 20)
            draw_debug("Edges", edges, 160)
            draw_debug("Masked", masked, 300)

            pygame.display.update()

        
        # Start image processing in separate thread
        processing_thread = Thread(target=camera_callback)
        processing_thread.start()

        camera.listen(lambda image: camera_callback(image))

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                world.tick()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            for actor in self.actors:
                if actor.is_alive:
                    actor.destroy()
            pygame.quit()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    carla_lane_detection = CarlaLaneDetection()
    carla_lane_detection.run()