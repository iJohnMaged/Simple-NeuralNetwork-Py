import sys, os
# a simple way to access neural_network module.
sys.path.append(f'..{os.sep}..')
from activation_fucntions import *
from neural_network import NeuralNetwork
import pygame
import random
import pygame_objects as po
import time


# XOR training data to feed into the NeuralNetwork

ONE = 1
ZERO = 0


TRAINING_DATA = [
    {
        'input': [0, 0],
        'output': [ZERO]
    },
    {
        'input': [0, 1],
        'output': [ONE]
    },
    {
        'input': [1, 0],
        'output': [ONE]
    },
    {
        'input': [1, 1],
        'output': [ZERO]
    }
]

# RGB Values for colors.

DARKGRAY = '#A9A9A9'

# Useful constant values

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 500

CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

FPS = 60

RESOLUTION = 10

# Initialize PyGame
pygame.init()
font = pygame.font.SysFont("Verdana", 12)
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

po.font = font
po.screen = screen


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]

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
    # Starts with a value of 0.01 and has a range of 0.01 to 0.5
    lr_slider = po.Slider("LR", 0.01, 0.5, 0.01, 150, 400)

    # A button to reset the NN when pressed.
    reset_button = po.Button("Reset NN!", (200, 470), None)
    save_button = po.Button("Save SS", (325, 470), None)

    # A Neural Network object with 2 inputs, 2 hidden nodes and 1 output.
    nn = NeuralNetwork(2, 2, 1, lr=lr_slider.val, activation_func=SIGMOID)

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
                    nn.reset()
                elif save_button.rect.collidepoint(pos):
                    # Save the current canvas
                    pygame.image.save(canvas, f'NN_{time.strftime("%Y%m%d-%H%M%S")}.jpg')


            elif event.type == pygame.MOUSEBUTTONUP:
                lr_slider.hit = False

        if lr_slider.hit:
            lr_slider.move()

        # Updating the learning rate every iteration with the value from slider.
        nn.lr = lr_slider.val

        # Training the neural network 100 times per frame using a random data point from the dataset.
        for _ in range(1000):
            data_point = random.choice(TRAINING_DATA)
            nn.train(data_point['input'], data_point['output'])

        # A pixels array used to visualize the output of the Neural Network
        # p = np.zeros((CANVAS_WIDTH, CANVAS_HEIGHT))

        for i in range(cols):
            for j in range(rows):
                # Calculate point (x, y) of the canvas to feed forward into the NN to get the expected XOR output
                # of this point.
                x = i / cols
                y = j / rows
                output = nn.predict([x, y])
                # Using equal values of (r, g, b) for a color
                rgb = int(output[0, 0] * 255)
                # rgb = max(min(int(output[0, 0] * 255), 255), 0)
                a = i*RESOLUTION
                b = j*RESOLUTION
                # Coloring the pixels shaping the square of side size (RESOLUTION) start at (x, y) with a color value
                # proportional to the output (white for 1 and black for 0)
                pygame.draw.rect(canvas, (rgb, rgb, rgb), [a, b, RESOLUTION, RESOLUTION])

        screen.blit(canvas, (0, 0))
        screen.blit(base, (0 , 400))
        lr_slider.draw()
        reset_button.draw()
        save_button.draw()
        pygame.display.flip()


if __name__ == '__main__':
    main()