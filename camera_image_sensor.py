import sys
import time
import os
import shutil
import glob
import random
import queue
import detection_yolo as dyolo

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def cleanup_output_directory(output_dir):
    # Delete any existing files in the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Cleaned up and prepared output directory: {output_dir}")


def main():

    # ---------- Connect to the CARLA Server ---------- #
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # ---------- Load world and all available blueprints ---------- #
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Find a cybertruck blueprint
    vehicle_bp = blueprint_library.find("vehicle.tesla.cybertruck") 
    
    # Choose a spawn point for the car
    spawn_points = world.get_map().get_spawn_points()
    vehicle_spawn_point = random.choice(spawn_points)
    
    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, vehicle_spawn_point)
    print("Vehicle spawned.")

    # Retrieve vehicle dimensions
    vehicle_extent = vehicle.bounding_box.extent
    vehicle_length = vehicle_extent.x * 2  # Full length of the vehicle
    vehicle_width = vehicle_extent.y * 2   # Full width of the vehicle
    vehicle_height = vehicle_extent.z * 2  # Full height of the vehicle

    print(f"Vehicle dimensions - Length: {vehicle_length} m, Width: {vehicle_width} m, Height: {vehicle_height} m")

    # Enable autopilot
    vehicle.set_autopilot(True)
    print("Autopilot enabled.")

    # Find a camera blueprint
    camera_bp = blueprint_library.find('sensor.camera.rgb')

    # Adjust camera attributes (optional)
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '105') # Field of view in degrees
    camera_bp.set_attribute('sensor_tick', '1.0') # Capture image every second

     # Calculate camera location based on vehicle dimensions
    camera_location = carla.Location(
        x=vehicle_extent.x + 0.5,    # Slightly forward from the vehicle's front
        y=0.0,                       # Centered horizontally on the vehicle
        z=vehicle_extent.z           # Elevated above the roof
    )

    # Adjust camera rotation to face forward
    camera_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)  # Tilt slightly downward
    camera_transform = carla.Transform(camera_location, camera_rotation)
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Spectator - Pointing to spawn point 0
    spectator = world.get_spectator()
    spectator_loc = carla.Location(x=vehicle_spawn_point.location.x, y=vehicle_spawn_point.location.y, z=vehicle_spawn_point.location.z+50.0)
    spectator_rot = carla.Rotation(pitch=-69.0, yaw=0.0, roll=0.0)
    spectator_trans = carla.Transform(spectator_loc, spectator_rot)

    # Move the spectator
    spectator.set_transform(spectator_trans)

    # Print the current working directory (where files will be saved)
    print("Saving files to:", os.getcwd())

    # Create directory for saving images if it doesn't exist
    output_dir = "image_data"
    cleanup_output_directory(output_dir)

    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)

    # Start listening to camera images
    camera.listen(lambda image: image.save_to_disk(f'{output_dir}/{image.frame}.png'))

    try:
        # # Let the simulation run for 30 seconds
        time.sleep(60*1)
    
    finally:
        # Clean up by destroying actors
        camera.stop()
        camera.destroy()
        vehicle.destroy()
    print("Actors destroyed, simulation finished.")

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')