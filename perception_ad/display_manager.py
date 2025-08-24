import pygame
from typing import Dict, List, Tuple
import cv2
import numpy as np

class DisplayManager:
    """
    Handles all visualization of sensors through the pygame display
    """
    def __init__(self, config: Dict):
        # Initialize pygame
        pygame.init()
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), 16)
        self.pygame_display = (
            config['pygame_display']['display_width'], 
            config['pygame_display']['display_height']
        )
        self.pygame_display_width, self.pygame_display_height = self.pygame_display
        self.display = pygame.display.set_mode(self.pygame_display)
        pygame.display.set_caption("CARLA Synchronous Client")

        self.grid_size = config['grid_size']
        self.grid_rows, self.grid_cols = self.grid_size
        self.cell_width = self.pygame_display_width // self.grid_cols
        self.cell_height = self.pygame_display_height // self.grid_rows

        self.sensor_grid = {} # sensor_name -> (grid_row, grid_col)
        self.sensor_images = {}  # sensor_name -> pygame.Surface

    def add_sensor(self, sensor_name, grid_position):
        """Register a sensor for display at specific grid position"""
        grid_row, grid_col = grid_position

        # Check for valid grid positions
        if not (0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols):
            raise ValueError(f"Grid position [{grid_position}] out of bounds for a {self.grid_rows}x{self.grid_cols} grid.")
        
        self.sensor_grid[sensor_name] = grid_position
        print(f"Registered sensor '{sensor_name}' at grid position {grid_position}")
    
    def update_sensor_image(self, sensor_name, image):
        """Update the image for a specific sensor"""
        self.sensor_images[sensor_name] = image

    def render(self):
        """Render all sensor images to the grid"""
        for sensor_name, grid_pos in self.sensor_grid.items():
            if sensor_name in self.sensor_images.keys():
                display_image = self.sensor_images[sensor_name]
                resize_display_image = self._scale_image_to_cell(display_image, (self.cell_width, self.cell_height))
                if self.render_enabled():
                    surface = pygame.surfarray.make_surface(np.rot90(np.fliplr(resize_display_image)))
                    self.display.blit(surface, self.get_display_offset(grid_pos))
    
    # Create debug miniatures for development and debugging
    def draw_debug(self, title, img, y_offset):
        debug_img = cv2.resize(img, (160, 120))
        if len(debug_img.shape) == 2:  # Convert grayscale to RGB for display
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
        debug_surface = pygame.surfarray.make_surface(np.rot90(np.fliplr(debug_img)))
        self.display.blit(debug_surface, (self.pygame_display_width-200, y_offset))
        
        # Add text label for clarity
        text = self.font.render(title, True, (255, 255, 255))
        self.display.blit(text, (self.pygame_display_width-200, y_offset))
    
    def _scale_image_to_cell(self, image, target_cell_size):
        """Scale image to fit cell while maintaining aspect ratio"""
        cell_width, cell_height = target_cell_size
        img_height, img_width = image.shape[:2]
        
        # Calculate scale factor (fit within cell, don't exceed)
        scale_width = cell_width / img_width
        scale_height = cell_height / img_height  
        scale = min(scale_width, scale_height)  # Use smaller scale to fit both dimensions
        
        # New dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create cell-sized surface with black background
        cell_surface = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        
        # Center the resized image
        x_offset = (cell_width - new_width) // 2
        y_offset = (cell_height - new_height) // 2
        
        cell_surface[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return cell_surface

    def cleanup_pygame(self):
        """Safe pygame cleanup"""
        try:
            pygame.quit()
        except:
            pass

    def render_enabled(self):
        return self.display != None
    
    def get_display_offset(self, gridPos):
        return [int(gridPos[1] * self.cell_width), int(gridPos[0] * self.cell_height)]

    def get_sensor_list(self) -> List:
        return list(self.sensor_grid.keys())

    def get_full_display_size(self) -> Tuple[int, int]:
        return self.pygame_display
    
    def get_cell_display_size(self) -> Tuple[int, int]:
        return [self.cell_width, self.cell_height]
    
    def get_grid_size(self) -> Tuple[int, int]:
        return self.grid_size