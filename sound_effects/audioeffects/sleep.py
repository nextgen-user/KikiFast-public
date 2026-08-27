import math
import time
import pygame
from pygame.locals import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
sys.path.insert(1, '/srv/kikifast/memory')
from memory import Memory
m = Memory()
import subprocess
m.update_data('currentmode', 'sleepy')
m.save()
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_rect().size
pygame.display.set_caption("ROBOTIC EYES")

# Original OLED dimensions
width = 128
height = 32

# Create PIL image and drawing context
image = Image.new("RGB", (width, height), (0, 0, 0))  # Use RGB mode for color
draw = ImageDraw.Draw(image)
color_light = (170,170,170)
# Eye color (blue)
eye_color = (39, 139, 244)  # RGB: blue

# Add stop button variables
button_font = pygame.font.Font(None, 36)
button_color = (200, 50, 50)
button_hover_color = (220, 80, 80)
button_rect = pygame.Rect(screen_width - 150-100, screen_height - 50, 140-70, 40)
button_text = button_font.render("MODE", True, (255, 255, 255))
button_hover = False

# Add AI text and divider variables
# We will use a smaller font for the long paragraph
ai_paragraph_font = pygame.font.Font("/srv/kikifast/audioeffects/Laconic_Regular.otf", 25) # Smaller font for the paragraph
ai_text_color = (255, 255, 255) # White color

# The new Lorem Ipsum paragraph
lorem_ipsum_text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum"

# Function to wrap text based on a maximum width
def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line_words = []
    
    for word in words:
        # Test if adding the next word exceeds max_width
        test_line = ' '.join(current_line_words + [word])
        
        # Check if the current line with the new word exceeds the max_width
        # or if the word itself is longer than max_width (should go on its own line)
        if font.size(test_line)[0] <= max_width:
            current_line_words.append(word)
        else:
            # If current_line_words is empty, it means the single 'word' itself is too long
            # In that case, we add the word as its own line (it will exceed max_width if it's super long)
            if not current_line_words:
                lines.append(word) 
                current_line_words = []
            else:
                # Add the accumulated words as a line
                lines.append(' '.join(current_line_words))
                # Start a new line with the current word
                current_line_words = [word]
    
    # Add the last line if it's not empty
    if current_line_words:
        lines.append(' '.join(current_line_words))
        
    return lines

# Main rendering function
def sleepy():

    # Clear the image with black background
    draw.rectangle((0, 0, width, height), outline=(0, 0, 0), fill=(0, 0, 0))
    
    # Draw shapes
    padding = 1
    shape_width = 32
    top = padding -10
    bottom = height - padding -30
    
    # First rectangle (eye)
    x = 20
    draw.rectangle((x, top + 10, x + shape_width, bottom), 
                   outline=eye_color, fill=eye_color)  # Blue eye
    
    # Second rectangle (eye)
    x = 80
    draw.rectangle((x, top + 10, x + shape_width, bottom), 
                   outline=eye_color, fill=eye_color)  # Blue eye
    
    # Convert PIL image to Pygame surface
    pil_str = image.tobytes()
    pygame_surface = pygame.image.fromstring(pil_str, (width, height), "RGB")
    
    # Scale to fullscreen while maintaining aspect 
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
    # pygame.display.flip() # Removed here, done once at the end of the main loop

# Main loop
running = True
start_time = time.time()

def awake():
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
    draw.rectangle((x, top-10, x + shape_width, top + eye_height-10), 
                   outline=eye_color, fill=eye_color)  # Blue eye
    
    # Second rectangle (eye)
    x = 80
    draw.rectangle((x, top-10, x + shape_width, top + eye_height-10), 
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
    # pygame.display.flip() # Removed here, done once at the end of the main loop

def add_text():
    divider_y = screen_height - 300 

    # Draw white horizontal divider
    pygame.draw.line(screen, ai_text_color, (0, divider_y), (screen_width, divider_y), 2) # 2 pixels thick

    # Prepare and display the Lorem Ipsum text below the divider
    text_margin_x = 20 # Margin from left/right edges of the screen
    max_text_width = screen_width - (2 * text_margin_x)

    wrapped_lines = wrap_text(lorem_ipsum_text, ai_paragraph_font, max_text_width)

    # Starting Y position for the first line of text (below the divider)
    text_start_y = divider_y + 10 # 10 pixels below the divider

    for line in wrapped_lines:
        line_surface = ai_paragraph_font.render(line, True, ai_text_color)
        # Center text horizontally
        line_rect = line_surface.get_rect(center=(screen_width // 2, text_start_y + line_surface.get_height() // 2))
        screen.blit(line_surface, line_rect)
        text_start_y += line_surface.get_height() + 2 # Move down for the next line, with 2 pixels spacing

    pygame.display.flip()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False
        # Handle mouse events for stop button
        elif event.type == MOUSEMOTION:
            button_hover = button_rect.collidepoint(event.pos)
        elif event.type == MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                subprocess.Popen("sudo pkill mpv",shell=True)
                m.update_data('song', 'false')
                m.save()
    
    # Re-fill the screen with black at the start of each frame before drawing anything
    screen.fill((0, 0, 0)) 

    # Render display based on current mode
    time.sleep(1) # This sleep makes the updates happen every second
    m = Memory()

    if m.get_data('currentmode') == 'sleepy':
        t=time.time()

        sleepy()
        print(time.time()-t)
        print("sleepy")
    elif m.get_data('currentmode') == 'awake':
        awake()
        print("awake")
    
    # Draw stop button (always on bottom right)
    if m.get_data('song')== 'true':
        button_color_current = button_hover_color if button_hover else button_color
        pygame.draw.rect(screen, button_color_current, button_rect, border_radius=8)
        pygame.draw.rect(screen, (150, 30, 30), button_rect, 2, border_radius=8)  # Darker border
        text_rect = button_text.get_rect(center=button_rect.center)
        screen.blit(button_text, text_rect)
    add_text()


pygame.quit()