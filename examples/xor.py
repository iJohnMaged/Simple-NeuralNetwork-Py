import sys
# a simple way to access neural_network module.
sys.path.append('..')
from neural_network import NeuralNetwork
import pygame
import random
import numpy as np


# XOR training data to feed into the NeuralNetwork

TRAINING_DATA = [
    {
        'input': [0, 0],
        'output': [0]
    },
    {
        'input': [0, 1],
        'output': [1]
    },
    {
        'input': [1, 0],
        'output': [1]
    },
    {
        'input': [1, 1],
        'output': [0]
    }
]


# RGB Values for colors.

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
ORANGE = (200, 100, 50)
TRANS = (1, 1, 1)
DARKGRAY = '#A9A9A9'

# Useful constant values

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 500

CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

FPS = 60

RESOLUTION = 20

# Initialize PyGame
pygame.init()
font = pygame.font.SysFont("Verdana", 12)
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


class Button():
    """
    A class to create a pygame button

    This class was adapted from:
    https://www.dreamincode.net/forums/topic/401541-buttons-and-sliders-in-pygame/
    """
    def __init__(self, txt, location, action, bg=WHITE, fg=BLACK, size=(80, 30), font_name="Segoe Print", font_size=16):
        self.color = bg  # the static (normal) color
        self.bg = bg  # actual background color, can change on mouseover
        self.fg = fg  # text color
        self.size = size

        self.font = pygame.font.SysFont(font_name, font_size)
        self.txt = txt
        self.txt_surf = self.font.render(self.txt, 1, self.fg)
        self.txt_rect = self.txt_surf.get_rect(center=[s//2 for s in self.size])

        self.surface = pygame.surface.Surface(size)
        self.rect = self.surface.get_rect(center=location)

        self.call_back_ = action

    def draw(self):
        self.mouseover()

        self.surface.fill(self.bg)
        self.surface.blit(self.txt_surf, self.txt_rect)
        screen.blit(self.surface, self.rect)

    def mouseover(self):
        self.bg = self.color
        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(pos):
            self.bg = GREY  # mouseover color

    def call_back(self):
        self.call_back_()


class Slider:

    """
    A class to create a slider with a maximum and minimum value.

    This class was adapted from:
    https://www.dreamincode.net/forums/topic/401541-buttons-and-sliders-in-pygame/
    """
    def __init__(self, name, val, maxi, mini, xpos, ypos):
        self.name = name
        self.val = val  # start value
        self.maxi = maxi  # maximum at slider position right
        self.mini = mini  # minimum at slider position left
        self.xpos = xpos  # x-location on screen
        self.ypos = ypos
        self.surf = pygame.surface.Surface((100, 50))
        self.hit = False  # the hit attribute indicates slider movement due to mouse interaction

        self.txt_surf = font.render(f"{self.name}: {self.val:.2f}", 1, BLACK)
        self.txt_rect = self.txt_surf.get_rect(center=(50 ,15))

        # Static graphics - slider background #
        self.surf.fill((100, 100, 100))
        pygame.draw.rect(self.surf, GREY, [0, 0, 100, 50], 3)
        pygame.draw.rect(self.surf, WHITE, [10, 8.5, 80, 15], 0)
        pygame.draw.rect(self.surf, WHITE, [10, 30, 80, 5], 0)

        self.surf.blit(self.txt_surf, self.txt_rect)  # this surface never changes

        # dynamic graphics - button surface #
        self.button_surf = pygame.surface.Surface((20, 20))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.circle(self.button_surf, BLACK, (10, 10), 6, 0)
        pygame.draw.circle(self.button_surf, ORANGE, (10, 10), 4, 0)

    def draw(self):
        """ Combination of static and dynamic graphics in a copy of
    the basic slide surface
    """
        # static
        surf = self.surf.copy()

        # dynamic
        pos = (10+int((self.val-self.mini)/(self.maxi-self.mini)*80), 33)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)  # move of button box to correct screen position

        self.txt_surf = font.render(f"{self.name}: {self.val:.2f}", 1, BLACK)
        pygame.draw.rect(self.surf, WHITE, [10, 8.5, 80, 15], 0)

        self.surf.blit(self.txt_surf, self.txt_rect)

        # screen
        screen.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """
        The dynamic part; reacts to movement of the slider button.
        """
        self.val = (pygame.mouse.get_pos()[0] - self.xpos - 10) / 80 * (self.maxi - self.mini) + self.mini
        if self.val < self.mini:
            self.val = self.mini
        if self.val > self.maxi:
            self.val = self.maxi


def main():

    # The canvas where the XOR problem visualization is shown
    canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
    canvas.fill(pygame.Color('#FFFFFF'))

    # A base surface used for controls (Currently only LR slider.)
    base = pygame.Surface((400, 100))
    base.fill(pygame.Color(DARKGRAY))

    # Divide the canvas into rows and cols with size of RESOLUTION
    rows = CANVAS_HEIGHT // RESOLUTION
    cols = CANVAS_WIDTH // RESOLUTION

    clock = pygame.time.Clock()

    # A slider to control the learning rate value in the neural network.
    # Starts with a value of 0.05 and has a range of 0.01 to 0.5
    lr_slider = Slider("LR", 0.05, 0.5, 0.01, 150, 400)

    # A button to reset the NN when pressed.
    reset_button = Button("Reset NN!", (200, 470), None)

    nn = NeuralNetwork(2, 2, 1, lr=lr_slider.val)
    # A Neural Network object with 2 inputs, 2 hidden nodes and 1 output.

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if mouse collides with the slider button / reset button
                pos = pygame.mouse.get_pos()
                if lr_slider.button_rect.collidepoint(pos):
                    lr_slider.hit = True
                elif reset_button.rect.collidepoint(pos):
                    # Reset the NN
                    nn = NeuralNetwork(2, 2, 1, lr=lr_slider.val)

            elif event.type == pygame.MOUSEBUTTONUP:
                lr_slider.hit = False

        if lr_slider.hit:
            lr_slider.move()

        # Updating the learning rate every iteration with the value from slider.
        nn.lr = lr_slider.val

        # Training the neural network 100 times per frame using a random data point from the dataset.
        for _ in range(100):
            data = random.choice(TRAINING_DATA)
            nn.train(data['input'], data['output'])

        # A pixels array used to visualize the output of the Neural Network
        p = np.zeros((CANVAS_WIDTH, CANVAS_HEIGHT))
        for i in range(cols):
            for j in range(rows):
                # Calculate point (x, y) of the canvas to feed forward into the NN to get the expected XOR output
                # of this point.
                x = i / cols
                y = j / rows
                output = nn.predict([x, y])
                # Using equal values of (r, g, b) for a color
                rgb = int(255 * output[0])
                # Converting RGB value to decimal.
                color = (rgb << 16) + (rgb << 8) + (rgb)
                a = i*RESOLUTION
                b = j*RESOLUTION
                # Coloring the pixels shaping the square of side size (RESOLUTION) start at (x, y) with a color value
                # proportional to the output (white for 1 and black for 0)
                p[a:a+RESOLUTION, b:b+RESOLUTION] = color

        pygame.surfarray.blit_array(canvas, p)
        screen.blit(canvas, (0, 0))
        screen.blit(base, (0 , 400))
        lr_slider.draw()
        reset_button.draw()
        pygame.display.flip()


if __name__ == '__main__':
    main()