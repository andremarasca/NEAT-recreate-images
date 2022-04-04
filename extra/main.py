from ast import While
from re import X
from xmlrpc.server import DocCGIXMLRPCRequestHandler
from pygame import MOUSEBUTTONDOWN
from pygame.locals import *
import pygame
import neat
from random import randint
import numpy as np
import os
import gzip
import pickle  # pylint: disable=import-error
from math import sin, pi

checkpoint_no = None   

################################# LOAD UP A BASIC WINDOW AND CLOCK #################################
pygame.init()
N_IMAGES = 2
PADDING_W, PADDING_H = 50, 50
IMAGE_W, IMAGE_H = 200, 200
DISPLAY_W, DISPLAY_H = PADDING_W * 3 + IMAGE_W * N_IMAGES, PADDING_H * 2 + IMAGE_H * 1
surface = pygame.Surface((DISPLAY_W,DISPLAY_H))
window = pygame.display.set_mode(((DISPLAY_W,DISPLAY_H)))
clock = pygame.time.Clock()
TARGET_FPS = 60
FONT_SIZE = 18
font = pygame.font.SysFont('arial', FONT_SIZE, True, False)
################################# GAME LOOP ##########################


def wait():
    pygame.event.clear()
    while True:
        event = pygame.event.wait()
        if event.type == QUIT:
            pygame.quit()
            exit()
        elif event.type == KEYDOWN:
            if event.key == K_a:
                return -1
            if event.key == K_s:
                return 1

# Metodo de comparação de algoritmo de ordenação
def compare(genome1, genome2, config):
    color = Color(0, 180, 240)
    surface.fill(color) # Fills the entire screen with light blue
    for img_no, gen in enumerate([genome1, genome2]):
        x_offset = (1 + img_no) * PADDING_W + img_no * IMAGE_W
        y_offset = PADDING_H
        net = neat.nn.FeedForwardNetwork.create(gen[1], config)
        text = font.render(f'Id: {gen[0]}', True, (0, 0, 0))
        surface.blit(text, (x_offset, y_offset-FONT_SIZE))
        for xx in range(IMAGE_W):
            for yy in range(IMAGE_H):
                x = x_offset + xx
                y = y_offset + yy

                X, Y = xx/IMAGE_W-0.5, yy/IMAGE_H-0.5
                X2, Y2 = X*X, Y*Y
                XY = X*Y
                SINX, SINY = sin(X*pi), sin(Y*pi)

                output = net.activate([X, Y, X2, Y2, XY, SINX, SINY])
                # color.hsla = (round(output[0]*360), round(output[1]*100), round(output[2]*100), 100)
                color.hsva = (round(output[0]*360), 100, 100, 100)
                # color = (round(output[0]*255), round(output[1]*255), round(output[2]*100), 255)
                surface.set_at((x, y), color)

    window.blit(surface, (0,0))
    pygame.display.flip()

    return wait()

def cmp_to_key(mycmp, config):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj, config) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj, config) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj, config) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj, config) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj, config) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj, config) != 0
    return K

def eval_genomes(genomes, config):
    specimens = sorted([*genomes], key=cmp_to_key(compare, config))
    tmp = 1 if not checkpoint_no else checkpoint_no
    fitness = tmp * len(genomes)
    for genome_id, genome in specimens:
        genome.fitness = fitness
        fitness -= 1

# Setup the NEAT Neural Network
def run(config_path, checkpoint_no):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create the population, which is the top-level object for a NEAT run.
    if not checkpoint_no:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % checkpoint_no)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    
    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 310)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path, checkpoint_no)