def cleanup_vehicles(vehicles):
        """Cleanup vehicles/traffic"""
        for vehicle in vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.set_autopilot(False)
                    vehicle.destroy()
            except:
                pass
        vehicles.clear()

def cleanup_sensors(sensors):
        """Cleanup cameras, lidars, etc."""
        for sensor in sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except:
                pass
        sensors.clear()