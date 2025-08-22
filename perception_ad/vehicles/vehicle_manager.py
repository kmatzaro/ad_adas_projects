import random
import carla
from actor_utils import cleanup_vehicles
from typing import Optional, Tuple


class VehicleSpawner():
    """
        Vehicle actor spawning with validation and retry logic.
        
        Actor spawning can fail due to:
        - Collision with existing actors
        - Invalid spawn points
    """
    def __init__(self, world):
        self.world = world
        self._blueprint_library = self.world.get_blueprint_library()
        self._spawn_points = self.world.get_map().get_spawn_points()
        self._vehicle_blueprints = self._blueprint_library.filter('vehicle.*')

    def spawn_vehicle(self, vehicle_type) -> Optional[carla.Actor]:
        """
        Actual vehicle spawn
        """
        try:
            if not self._spawn_points:
                raise RuntimeError("No spawn points available on map")
            
            print(f"Found {len(self._spawn_points)} spawn points")
            
            # Spawn vehicle with collision checking and retry logic
            if vehicle_type != 'random':
                vehicle_bp = self._vehicle_blueprints.find(f'vehicle.{vehicle_type}')[0]
            else:
                vehicle_bp = random.choice(self._vehicle_blueprints)
            
            for attempt in range(5):  # Try multiple spawn points
                try:
                    spawn_point = random.choice(self._spawn_points)
                    
                    # Check if spawn point is clear before attempting spawn
                    if self._is_spawn_point_clear(spawn_point):
                        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                        print(f"Vehicle spawned successfully at attempt {attempt + 1}")
                        break
                        
                except RuntimeError as e:
                    if "collision" in str(e).lower() and attempt < 4:
                        continue  # Try different spawn point on collision
                    else:
                        raise e
            else:
                raise RuntimeError("Could not find clear spawn point for vehicle")
        
            return vehicle
        
        except Exception as e:
            print(f"Vehicle actor spawning failed: {e}")
            return None
        
    def _is_spawn_point_clear(self, spawn_point, radius=2.0):
        """
        Check if spawn point is clear of other vehicles.
        
        Prevents collision errors during spawn by checking occupancy first.
        Much better than try/except on spawn because it's proactive.
        """
        location = spawn_point.location
        
        # Get all vehicles in the world to check for conflicts
        vehicles = self.world.get_actors().filter('vehicle.*')
        
        for vehicle in vehicles:
            if vehicle.get_location().distance(location) < radius:
                return False
        
        return True

class VehicleManager():
    """
    This class is responsible for spawning, configuring and destroying vehicle assests in simulation
    """
    def __init__(self, world):
        self.vehicle_spawner = VehicleSpawner(world)
        self.vehicle = None
        self.vehicle_type = 'tesla.model3'
        self.autopilot_enabled = True
        self.vehicle_actors = []

    def spawn_vehicle(self):
        self.vehicle = self.vehicle_spawner.spawn_vehicle(self.vehicle_type)
        if self.vehicle:
            self.enable_autopilot()
            self.vehicle_actors.append(self.vehicle)

    def enable_autopilot(self):
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.set_autopilot(True)
            self.autopilot_enabled = True
    
    def disable_autopilot(self):
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.set_autopilot(False)
            self.autopilot_enabled = False
    
    def toggle_autopilot(self):
        if self.autopilot_enabled:
            self.disable_autopilot()
        else:
            self.enable_autopilot()
    
    def cleanup(self):
        """Clean up whatever this manager created"""
        if self.vehicle_actors:
            cleanup_vehicles(self.vehicle_actors)

class TrafficManager:
    """
    This class is responsible for spawning, configuring and destroying all traffic group actors that
    spawn in bulk
    """

    def __init__(self, world, number_of_traffic_vehicles):
        self.vehicle_spawner = VehicleSpawner(world)
        self.number_of_traffic_vehicles = number_of_traffic_vehicles
        self.traffic_actors = []

    def spawn_traffic(self):
        """Spawn traffic"""
      
        # Spawn a few vehicles at random locations
        for i in range(self.number_of_traffic_vehicles):          
            # Pick random vehicle type and spawn point
            traffic_vehicle = self.vehicle_spawner.spawn_vehicle('random')
            if traffic_vehicle:
                traffic_vehicle.set_autopilot(True)
                self.traffic_actors.append(traffic_vehicle)
    
    def cleanup(self):
        """Clean up whatever this manager created"""
        if self.traffic_actors:
            cleanup_vehicles(self.traffic_actors)