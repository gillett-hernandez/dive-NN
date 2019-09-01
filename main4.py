#!/usr/bin/env python3
import argparse
# import numpy as np
from random import uniform as random
import random as rand
import time
from math import pi as PI
from math import cos, sin, atan2, atan, tanh, fmod, exp, isclose
from math import hypot as mag
import json
from itertools import takewhile
from namespace import Namespace

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_false")
parser.add_argument("--read-data", action="store_true")
parser.add_argument("--write-data", action="store_true")
parser.add_argument("--show-trails", action="store_true")
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--batches", type=int, default=100)
parser.add_argument("--scale", type=int, default=15)
parser.add_argument("--n-players", type=int, default=128)
parser.add_argument("--draw-vectors", action="store_true")
parser.add_argument("--debug", action="store_true")


args = parser.parse_args()

HEADLESS = args.headless
FRAMESKIP = args.frameskip
should_write_training_data = args.write_data
should_read_training_data = args.read_data
BATCHES = args.batches
N_PLAYERS = args.n_players
scale = args.scale
should_draw_vectors = args.draw_vectors
debug = args.debug

if not HEADLESS:
    import pygame
    from pygame.locals import *
    import colorsys


# import tensorflow as tf


HALF_PI = PI / 2
TAU = PI * 2

g = 9.80665
drag_flat = 0.07
drag_narrow = 0.038
mass = 100
influence = 900

MUTATION_EFFECT = 0.20
MUTATION_CHANCE = 0.20
timedelta = 0.1

N_PARAMS = 11
TTL = 1000
OFFSET = (100, 100)
RANDOM_LOWER_BOUND = 15000
RANDOM_UPPER_BOUND = 15000
PARAM_LOWER_BOUND = -1
PARAM_UPPER_BOUND = 1
RANDOM_INITIAL = 15000
FITNESS_HYPERPARAMETER_WIDTH=10000
FITNESS_HYPERPARAMETER_HEIGHT=30000
resources = Namespace()

offset = 0
text_to_render = []
longest_width = 0


def render_text(text, color=(0, 0, 0)):
    global offset
    global text_to_render
    global longest_width
    if HEADLESS:
        return
    if not isinstance(text, str):
        text = str(text)
    font = resources.font
    size = font.size(text)
    h = size[1]
    size = pygame.Rect((0, offset), size)
    offset += h
    if size[0] > longest_width:
        longest_width = size[0]
    text = font.render(text, False, color)
    text_to_render.append((text, size))


def clear_text_buffer():
    global offset
    global text_to_render
    global longest_width
    longest_width = 0
    offset = 0
    text_to_render = []


def crossover(brain1, brain2):
    # bits are basically the binary choices for which parameter to take from which parent
    if hasattr(rand, "choices"):
        bits = rand.choices([0, 1], k=N_PARAMS)
    else:
        bits = [rand.choice([0, 1]) for _ in range(N_PARAMS)]
    new_brain = [[brain1[i], brain2[i]][bit] for i, bit in enumerate(bits)]
    return new_brain


def mutation(brain):
    for i, param in enumerate(brain):
        if random(0, 1) < MUTATION_CHANCE:
            # print("mutation occurred!")
            # brain.brain.params[i] = random(-MUTATION_EFFECT, MUTATION_EFFECT) + param
            brain[i] = rand.gauss(0, 2 * MUTATION_EFFECT) + param


def generate_children(brains, n=float("inf")):
    i = 0
    while i < n:
        brain1, brain2 = rand.sample(brains, 2)
        new_brain = crossover(brain1, brain2)
        mutation(new_brain)
        yield new_brain
        i += 1

PROPORTIONS = [0.25, 0.25, 0.6, 0.15]
def selection_crossover_and_breeding(fitnesses, brains):
    new_brains = []
    # sort by fitness, lower is better
    zipped = list(zip(fitnesses, brains))
    zipped.sort(key=lambda e: e[0], reverse=True)
    brains = list(zip(*zipped))[1]
    # truncate X% worst players
    brains = brains[: int(PROPORTIONS[0]*N_PLAYERS)]
    # keep the Y% greatest brains
    new_brains.extend(brains[: int(PROPORTIONS[1]*N_PLAYERS)])

    # breed brains to fill Z% of the new population
    for new_brain in generate_children(brains, n=int(PROPORTIONS[2]*N_PLAYERS)):
        new_brains.append(new_brain)
    # fill the rest with new randoms
    while len(new_brains) < N_PLAYERS:
        new_brains.append(construct_brain())
    assert len(new_brains) == N_PLAYERS, (len(new_brains), N_PLAYERS)
    return new_brains


def sign(x):
    return 1 if x >= 0 else -1


def vadd(l1, l2):
    return l1[0] + l2[0], l1[1] + l2[1]


def vsub(l1, l2):
    return l1[0] - l2[0], l1[1] - l2[1]


def construct_brain():
    return [random(PARAM_LOWER_BOUND, PARAM_UPPER_BOUND) for _ in range(N_PARAMS)]


def remap(variables):
    return [
        variables[0], # x
        variables[1], # y
        pmag(variables), # mag
        direction(variables), # dir
        variables[4], # theta
        mag(variables[5]-variables[0], variables[6]-variables[1]), # distance to target
        atan2(variables[6]-variables[1], variables[5]-variables[0]), # angle to target
        variables[5]-variables[0], # horizontal distance to target
        variables[6]-variables[1], # vertical distance to target
        variables[7] # time
    ]

def evaluate(variables, params):
    rm = remap(variables)
    return fmod(
        params[-1] + sum(p * s for p, s in zip(params, remap(variables))),
        TAU
    )


def out_of_bounds(player):
    return player[0] < -100 or player[1] > player[6]

def direction(player):
    return atan2(player[3], player[2]) if player[2] != 0 else (PI+HALF_PI)

def AoA(player):
    # return atan2(player[3], player[2])
    return fmod(direction(player) - player[4], TAU)

def tangent(player):
    return ((atan2(-player[2],player[3]) + (PI if player[3] > 0 else 0)) if player[3] != 0 else HALF_PI) % TAU


def pmag(player):
    return mag(player[2], player[3])


def lift_force(player):
    # angle = (self.theta - self.direction) % TAU

    y = 0.7 * 1.225 * 0.75 / (2*mass)
    # normal lift
    _AoA = AoA(player)
    _tan = tangent(player)
    mul = 50*y * pmag(player)**2 * cos(_AoA) * sin(_AoA)
    return [mul*cos(_tan), mul*sin(_tan)]

def simulate(player):
    L = lift_force(player)
    _mag = mag(player[2], player[3])
    # f = drag_narrow + (drag_flat-drag_narrow) * abs(sin(self.AoA))
    f = drag_narrow
    player[2] += (L[0] - f * _mag * player[2])/mass
    player[3] += (L[1] - g * mass - f * _mag * player[3])/mass
    player[0] += player[2] * timedelta
    player[1] -= player[3] * timedelta
    player[7] += timedelta

def update(player, brain):
    player[4] = evaluate(player, brain)

def construct_player(target):
    return [
        0, # x
        0, # y
        0, # vx
        0, # vy
        0, # theta
        target[0], # target x
        target[1], # target y
        0, # time
        True, # alive
        0 # fitness
    ]


def reset(player):
    player[0:5] = [0,0,0,0,0]
    player[7] = 0
    player[-2] = True
    player[-1] = 0


def construct_player_and_brain(target, params=None):
    player = construct_player(target)
    if params is None:
        print("constructing brain")
        brain = construct_brain()
    else:
        print("constructing brain from params")
        brain = params[:]
    update(player, brain) 
    return player, brain

def construct_players_and_brains(DEST, filename=None):
    if should_read_training_data:
        print("reading training data")
        with open(filename, "r") as fd:
            data = json.load(fd)
        print(f"number of samples: {len(data['training_data'])}")

    players = []
    brains = []
    if should_read_training_data:
        print("constructing brain")
        for i in range(len(data["training_data"])):
            players.append(construct_player(DEST))
            brains.append(data["training_data"][i][:])
            update(players[-1], brains[-1])
    else:
        print("constructing brain from params")
        for i in range(N_PLAYERS):
            players.append(construct_player(DEST))
            brains.append(construct_brain())
            update(players[-1], brains[-1]) 
    return players, brains

def transform_pos(x=0, y=0):
    """returns the screen coordinates to draw pos"""
    return vadd(OFFSET, (int(x / scale), int(y / scale)))


def reverse_transform_position(x, y):
    """returns the approximate real coordinates that correspond to screen coordinates"""
    return vsub([e*scale for e in [x,y]], OFFSET)


def redraw_screen(screen, DEST, color1, color2, color3, color4):
    screen.fill(color1)

    # pygame.draw.rect(screen, color2, pygame.Rect(vadd(OFFSET, (0, 0)), list(int(e / scale) for e in DEST)))
    screen.blit(color2, vadd(OFFSET, (0,0)))
    pygame.draw.circle(screen, color3, vadd(OFFSET, (0, 0)), 5)
    pygame.draw.circle(screen, color4, vadd(OFFSET, list(int(e / scale) for e in DEST)), 5)


def player_fitness_formula(player):
    return fitness_formula(player[0], player[1], player[5], player[6], player[7])

def fitness_formula(x, y, tx, ty, time):
    return FITNESS_HYPERPARAMETER_HEIGHT * exp(-(mag(tx-x, ty-y) / FITNESS_HYPERPARAMETER_WIDTH) ** 2) - 2*time

assert fitness_formula(0,0,0,0,0) == 30000

def prepare_bg(tx, ty, SIZE, fitness_formula):
    surface = pygame.Surface((SIZE)).convert_alpha()
    for y in range(0, SIZE[1], 10):
        for x in range(0, SIZE[0], 10):
            nx, ny = reverse_transform_position(x, y)
            v = fitness_formula(nx, ny, tx, ty, 0)
            v = 1+exp(-1) - exp(-v/30000)
            # print(pos, v)
            surface.fill(pygame.Color(*[int(e*255) for e in colorsys.hls_to_rgb(v, 0.5, 0.5)]), (x,y, 10,10))
    return surface


def main(read_file="savedata.json", write_file = "savedata.json"):
    global HEADLESS
    if not HEADLESS:

        pygame.init()

    SIZE = WIDTH, HEIGHT = (1920, 1080)
    if not HEADLESS:
        screen = pygame.display.set_mode(SIZE)

    if not HEADLESS:
        BLACK = pygame.Color(0, 0, 0)
        WHITE = pygame.Color(255, 255, 255)
        GREEN = pygame.Color(0, 255, 0)
        RED = pygame.Color(255, 0, 0)
        GREY = pygame.Color(127, 127, 127)

    FLOOR = 10000
    DEST = RANDOM_INITIAL, FLOOR

    if not HEADLESS:
        WHITE_SURFACE = pygame.Surface((SIZE))
        WHITE_SURFACE.fill(WHITE)
        bg = prepare_bg(*DEST, vsub(transform_pos(*DEST), OFFSET), fitness_formula)

    if not HEADLESS:
        font = pygame.font.SysFont(pygame.font.get_default_font(), 22)
        resources.font = font

        redraw_screen(screen, DEST, WHITE, bg, GREEN, RED)

    players, brains = construct_players_and_brains(DEST, read_file)
    assert len(players) == len(brains)

    userPlayer = construct_player(DEST)
    userBrain = construct_brain()

    best_fitness = float("inf")

    for i in range(BATCHES):
        print(f"starting batch {i} of {BATCHES}")
        t1 = time.perf_counter()
        frame = 0
        if frame % FRAMESKIP == 0 and not HEADLESS:
            headless_flag = False
        else:
            headless_flag = True
        halted = False

        if not headless_flag:
            redraw_screen(screen, DEST, WHITE, bg, GREEN, RED)

        alive_players_count = len(players)
        while not halted:
            if frame % FRAMESKIP == 0 and not HEADLESS:
                headless_flag = False
            else:
                headless_flag = True
            if not headless_flag:

                for e in pygame.event.get():
                    if (e.type == KEYDOWN and e.key in [K_q, K_ESCAPE]) or e.type == QUIT:
                        return
                    elif e.type == KEYDOWN:
                        if e.key == K_r:
                            # reset canvas
                            if not headless_flag:
                                redraw_screen(screen, DEST, WHITE, bg, GREEN, RED)
                            for player in players:
                                player.reset()
                            userPlayer.reset()
                        elif e.key == K_x:
                            # exit and save
                            if should_write_training_data:
                                with open("save_data.json", "w") as fd:
                                    training_data = {"training_data": [player.brain.params for player in players]}
                                    json.dump(training_data, fd, indent=4)
                            pygame.quit()
                            return

            if not headless_flag and not args.show_trails:
                screen.fill(WHITE)
                redraw_screen(screen, DEST, WHITE, bg, GREEN, RED)

            render_text(f"fittest players momentum = {mass * mag(players[0][2], players[0][3])}")
            # render_text(f"user players momentum = {mass * mag(userPlayer.vx, userPlayer.vy)}")
            # render_text(f"user players speed = {1 * mag(userPlayer.vx, userPlayer.vy)}")
            # render_text(f"user players direction = {userPlayer.direction}")
            # render_text(f"user players theta = {userPlayer.theta}")
            # render_text(f"user players AoA = {userPlayer.AoA}")
            # render_text(f"user players velocity = {[userPlayer.vx, userPlayer.vy]}")
            # print(f"percentage of players dead = {len([p for p in players if not p[-2]])/len(players)}")
            # print(f"alive players count {alive_players_count}")
            for player, brain in zip(reversed(players), reversed(brains)):
                if not player[-2]:
                    continue
                simulate(player)
                update(player, brain)
                if player[-2] and out_of_bounds(player):
                    player[-2] = False # assign alive status
                    player[-1] = player_fitness_formula(player) # assign fitness
                    # print(player[-1])
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, transform_pos(player[0], player[1]), 3)
                        render_text(player[-1])
                    alive_players_count -= 1
                    continue
                if player[-2] and player[7] > TTL:
                    player[-2] = False # assign alive status
                    player[-1] = player_fitness_formula(player) # assign fitness
                    # print(player[-1])
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, transform_pos(player[0], player[1]), 3)
                        render_text(player[-1])
                    alive_players_count -= 1
                    continue
                if not headless_flag:
                    pygame.draw.circle(screen, BLACK, transform_pos(player[0], player[1]), 1)

            simulate(userPlayer)
            if not headless_flag:
                userPlayer[4] = -atan2(
                    *vsub([x * scale for x in vsub(pygame.mouse.get_pos()[::-1], OFFSET)], [userPlayer[1], userPlayer[0]])
                )
            else:
                # userPlayer.theta = PI / 2
                pass
            # print(userPlayer.theta)
            if userPlayer[-2] and out_of_bounds(userPlayer):
                userPlayer[-2] = False
            if userPlayer[-2] and userPlayer[7] > TTL:
                userPlayer[-2] = False
            if not headless_flag:
                pygame.draw.circle(screen, GREEN, transform_pos(userPlayer[0], userPlayer[1]), 1)
            if not headless_flag and should_draw_vectors:
                pygame.draw.line(
                    screen,
                    RED,
                    transform_pos(userPlayer[0], userPlayer[1]),
                    vadd(
                        [3 * influence * cos(userPlayer[4]) / mass, -3 * influence * sin(userPlayer[4]) / mass],
                        transform_pos(userPlayer[0], userPlayer[1]),
                    ),
                    1,
                )
                LF = lift_force(userPlayer)
                pygame.draw.line(
                    screen,
                    BLACK,
                    transform_pos(userPlayer[0], userPlayer[1]),
                    vadd(
                        [LF[0], -LF[1]],
                        transform_pos(userPlayer[0], userPlayer[1]),
                    ),
                    1,
                )
                pygame.draw.line(
                    screen,
                    BLACK,
                    transform_pos(userPlayer[0], userPlayer[1]),
                    vadd(
                        [userPlayer[2], -userPlayer[3]],
                        transform_pos(userPlayer[0], userPlayer[1]),
                    ),
                    1,
                )

            render_text(f"alive players = {alive_players_count}")

            if not headless_flag:
                screen.blit(WHITE_SURFACE, (0,0), (0,0, longest_width+500, offset))
                for text_surface, pos in text_to_render:
                    screen.blit(text_surface, pos)
            if alive_players_count == 0:
                halted = True

            if not headless_flag:
                pygame.display.flip()
            clear_text_buffer()
            frame += 1
        t2 = time.perf_counter()
        print(t2-t1)
        best_fitness = max(*[e[-1] for e in players])
        new_target = random(RANDOM_LOWER_BOUND, RANDOM_UPPER_BOUND), FLOOR
        print("new target = " + str(new_target))
        print(f"batch {i} of {BATCHES} = {round(100*i/BATCHES,2)}% done")
        print(f"best fitness was {best_fitness}")
        for player in players:
            player[5:7] = new_target
        print("performing genetic algorithm")
        brains = selection_crossover_and_breeding([p[-1] for p in players], brains)
        # assert len(set(id(e) for e in brains)) == len(brains)
        players = players[:len(brains)]
        print("resetting players")
        print("player 0 before reset", players[0])
        for player in players:
            reset(player)
        print("player 0 after reset", players[0])
        print("done resetting players")
        DEST = new_target
        reset(userPlayer)
        t3 = time.perf_counter()
        print(t3-t2)
    if should_write_training_data:
        with open(write_file, "w") as fd:
            training_data = {"training_data": brains}
            json.dump(training_data, fd, indent=4)
            # fd.write("{\n")
            # for player in players:
            #     fd.write("\t[" + ", ".join(str(e) for e in player.brain.params) + "],\n")
            # fd.write("}\n")


if __name__ == "__main__":
    main()
