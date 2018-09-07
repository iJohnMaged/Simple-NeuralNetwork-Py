"""
This module contains very useful objects (Buttons and sliders!) to work with pygame.

Source: https://www.dreamincode.net/forums/topic/401541-buttons-and-sliders-in-pygame/
"""


import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
ORANGE = (200, 100, 50)
TRANS = (1, 1, 1)
DARKGRAY = '#A9A9A9'

screen = None
font = None

class Button():
    """
    A class to create a pygame button

    This class was adapted from:
    https://www.dreamincode.net/forums/topic/401541-buttons-and-sliders-in-pygame/
    """
    def __init__(self, txt, location, action, bg=WHITE, fg=BLACK, size=(100, 30), font_name="Segoe Print", font_size=16):
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
    A class to create a pygame slider with a maximum and minimum value.

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
