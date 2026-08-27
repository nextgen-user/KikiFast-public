import math
import time
import pygame
from pygame.locals import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_rect().size
pygame.display.set_caption("OLED Simulator")

# Original OLED dimensions
width = 128
height = 32

# Create PIL image and drawing context
image = Image.new("RGB", (width, height), (0, 0, 0))  # Use RGB mode for color
draw = ImageDraw.Draw(image)

# Eye color (blue)
eye_color = (0, 0, 255)  # RGB: blue

# Main rendering function
def render():
    # Clear the image with black background
    draw.rectangle((0, 0, width, height), outline=(0, 0, 0), fill=(0, 0, 0))
    
    # Draw shapes
    padding = 1
    shape_width = 32
    eye_height = 20  # Increased height for more visible eyes
    
    # Calculate vertical position to center the eyes
    top = (height - eye_height) // 2
    
    # First rectangle (eye)
    x = 20
    draw.rectangle((x, top, x + shape_width, top + eye_height), 
                   outline=eye_color, fill=eye_color)  # Blue eye
    
    # Second rectangle (eye)
    x = 80
    draw.rectangle((x, top, x + shape_width, top + eye_height), 
                   outline=eye_color, fill=eye_color)  # Blue eye
    
    # Convert PIL image to Pygame surface
    pil_str = image.tobytes()
    pygame_surface = pygame.image.fromstring(pil_str, (width, height), "RGB")
    
    # Scale to fullscreen while maintaining aspect ratio
    scale_factor = min(screen_width / width, screen_height / height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    pos_x = (screen_width - new_width) // 2
    pos_y = (screen_height - new_height) // 2
    
    scaled_surface = pygame.transform.scale(
        pygame_surface, 
        (new_width, new_height)
    )
    
    # Render to screen
    screen.fill((0, 0, 0))  # Black background
    screen.blit(scaled_surface, (pos_x, pos_y))
    pygame.display.flip()

# Main loop
running = True
start_time = time.time()
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False
    
    # Render display
    render()
    
    # Exit after 1.5 seconds (like original)
    if time.time() - start_time >= 1500:
        running = False

pygame.quit()