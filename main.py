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
from PIL import Image
from math import sqrt, sin, pi

################################# LOAD UP A BASIC WINDOW AND CLOCK #################################
checkpoint_no = None
imprimir = False  
if imprimir: pygame.init()
N_IMAGES = 2
PADDING_W, PADDING_H = 50, 50
IMAGE_W, IMAGE_H = 200, 200
DISPLAY_W, DISPLAY_H = PADDING_W * 3 + IMAGE_W * N_IMAGES, PADDING_H * 2 + IMAGE_H * 1
if imprimir: surface = pygame.Surface((DISPLAY_W,DISPLAY_H))
if imprimir: window = pygame.display.set_mode(((DISPLAY_W,DISPLAY_H)))
if imprimir: clock = pygame.time.Clock()
TARGET_FPS = 60
FONT_SIZE = 18
if imprimir: font = pygame.font.SysFont('arial', FONT_SIZE, True, False)
################################# IMAGE ##########################
img = Image.open('Moon.jpg')
pix = img.load()
if imprimir: background_image = pygame.image.load('Moon.jpg').convert()
################################# GAME LOOP ##########################

best_fitness = -1

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
def eval_genome(gen, config, save_image_best=False):
    global best_fitness
    dot = 0
    u_norm = 0
    v_norm = 0
    img_no = 0
    color = Color(0, 180, 240)
    if imprimir: surface.fill(color)
    x_offset = (1 + img_no) * PADDING_W + img_no * IMAGE_W
    y_offset = PADDING_H
    net = neat.nn.FeedForwardNetwork.create(gen[1], config)
    if imprimir: text = font.render(f'Id: {gen[0]}', True, (0, 0, 0))
    if imprimir: surface.blit(text, (x_offset, y_offset-FONT_SIZE))
    if save_image_best: image = []
    for yy in range(IMAGE_H):
        if save_image_best: row = []
        for xx in range(IMAGE_W):
            x = x_offset + xx
            y = y_offset + yy

            X, Y = xx/IMAGE_W-0.5, yy/IMAGE_H-0.5
            X2, Y2 = X**2, Y**2
            # XY = X*Y
            # SINX = sin(X*pi)
            # SINY = sin(Y*pi)
            # output = net.activate([X, Y, X2, Y2, XY, SINX, SINY])
            output = net.activate([X2, Y2])

            color_pred = int(round((output[0] + 1.0) * 255 / 2.0))
            color_pred = max(0, min(255, color_pred))
            color_ref = (pix[xx, yy][0]+pix[xx, yy][1]+pix[xx, yy][2]) / 3
            dot += color_pred * color_ref
            u_norm += color_pred ** 2
            v_norm += color_ref ** 2

            if imprimir: surface.set_at((x, y), (color_pred, color_pred, color_pred))
            if save_image_best: row.append(color_pred)
        if save_image_best: image.append(row)

    img_no = 1
    x_offset = (1 + img_no) * PADDING_W + img_no * IMAGE_W
    y_offset = PADDING_H
    if imprimir: surface.blit(background_image, (x_offset, y_offset))
    if imprimir: window.blit(surface, (0,0))
    if imprimir: pygame.display.flip()

    fitness = 0
    u_norm = sqrt(u_norm)
    v_norm = sqrt(v_norm)
    if u_norm != 0 and v_norm != 0:
        fitness = dot / (u_norm * v_norm)       

    print(f'Id: {gen[0]} -- Fitness: {fitness}')

    if best_fitness < fitness:
        if save_image_best:
            im = np.clip(np.array(image), 0, 255).astype(np.uint8)
            im = Image.fromarray(im, mode="L")
            im.save(f'images/best_{int(fitness*1e5)}.png')
        else:
            eval_genome(gen, config, True)
        
        with gzip.open(f'bests/best_{int(fitness*1e5)}', 'w', compresslevel=5) as f:
            pickle.dump(gen, f, protocol=pickle.HIGHEST_PROTOCOL)

        best_fitness = fitness

    return fitness

def eval_genomes(genomes, config):
    for genome in genomes:
        # inputs()
        genome[1].fitness = eval_genome(genome, config)

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
    winner = p.run(eval_genomes, 9999)


def show_best(config_path, file_name):
    if imprimir == False:
        print("Mude Imprimir para True")
        return

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    with gzip.open(f'bests/{file_name}') as f:
        genome_best = pickle.load(f)

    eval_genome(genome_best, config)
    wait()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path, checkpoint_no)
    # show_best(config_path, 'best_56')