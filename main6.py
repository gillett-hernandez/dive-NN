#!/usr/bin/env python3
import argparse
# import numpy as np
from random import uniform as random
import random as rand
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


def crossover(flyer1, flyer2):
    # bits are basically the binary choices for which parameter to take from which parent
    if hasattr(rand, "choices"):
        bits = rand.choices([0, 1], k=N_PARAMS)
    else:
        bits = [rand.choice([0, 1]) for _ in range(N_PARAMS)]
    params1 = flyer1.brain.params
    params2 = flyer2.brain.params
    new_flyer = Player(flyer1.target)
    new_flyer.brain.params = [[params1[i], params2[i]][bit] for i, bit in enumerate(bits)]
    return new_flyer


def mutation(flyer):
    for i, param in enumerate(flyer.brain.params):
        if random(0, 1) < MUTATION_CHANCE:
            # print("mutation occurred!")
            # flyer.brain.params[i] = random(-MUTATION_EFFECT, MUTATION_EFFECT) + param
            flyer.brain.params[i] = rand.gauss(0, 2 * MUTATION_EFFECT) + param


def generate_children(players, n=float("inf")):
    i = 0
    while i < n:
        flyer1, flyer2 = rand.sample(players, 2)
        new_flyer = crossover(flyer1, flyer2)
        mutation(new_flyer)
        yield new_flyer
        i += 1

PROPORTIONS = [0.25, 0.25, 0.6, 0.15]
def selection_crossover_and_breeding(flyers):
    new_flyers = []
    # sort by fitness, lower is better
    flyers.sort(key=lambda e: e.fitness, reverse=True)
    # truncate X% worst flyers
    flyers = flyers[: int(PROPORTIONS[0]*N_PLAYERS)]
    # keep the Y% greatest flyers
    new_flyers.extend(flyers[: int(PROPORTIONS[1]*N_PLAYERS)])
    # shuffle flyers so we can split them into random batches
    rand.shuffle(flyers)
    # breed flyers to fill Z% of the new population
    for new_flyer in generate_children(flyers, n=int(PROPORTIONS[2]*N_PLAYERS)):
        new_flyers.append(new_flyer)
    # fill the rest with new randoms
    while len(new_flyers) < N_PLAYERS:
        new_flyers.append(Player(new_flyers[-1].target))
    assert len(new_flyers) == N_PLAYERS, (len(new_flyers), N_PLAYERS)
    return new_flyers


def sign(x):
    return 1 if x >= 0 else -1


def vadd(l1, l2):
    return l1[0] + l2[0], l1[1] + l2[1]


def vsub(l1, l2):
    return l1[0] - l2[0], l1[1] - l2[1]


def construct_players(DEST, filename):
    if should_read_training_data:
        with open(filename, "r") as fd:
            data = json.load(fd)

    flyers = []
    if should_read_training_data:
        for i in range(len(data["training_data"])):
            flyers.append(Player(DEST))
            flyers[-1].brain.params = data["training_data"][i]
    else:
        for i in range(N_PLAYERS):
            flyers.append(Player(DEST))
    return flyers


class Brain:
    __slots__ = ["model"]

    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(7, input_shape=(7,)),  # must declare input shape
            tf.keras.layers.Dense(1)
        ])

        # self.number_of_params = len(self.layer1) + len(self.bias1) + len(self.output) + len(self.bias_output)

        # self.bias = random(-2,2)

    def evaluate(self, player):
        stats = np.matrix([
            player.x,
            player.y,
            mag(player.vx, player.vy),
            atan2(player.vy,player.vx),
            player.theta,
            mag(*vsub(player.target, [player.x, player.y])),
            atan2(*vsub(player.target, [player.x, player.y]))
        ])
        # # print(stats, stats.shape)
        # output1 = (self.layer1 * stats + self.bias1).T
        # # print(output1.shape)
        # output2 = (self.bias_output + output1 * self.output)
        # print(output2.shape)
        # S = fmod(output2[0], TAU)
        S = fmod(self.model.predict(stats)[0], TAU)


class Player:
    __slots__ = ["x", "y", "vx", "vy", "theta", "brain", "alive", "target", "time", "fitness"]

    def __init__(self, target, params=None):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        # self.intent = 0
        self.theta = 0
        self.time = 0
        self.brain = Brain(self)
        if params is not None:
            self.brain.params = params
        self.target = target
        self.update()
        self.alive = True
        self.fitness = None

    @property
    def direction(self):
        return atan2(self.vy,self.vx) if self.vx != 0 else (PI+HALF_PI)

    @property
    def AoA(self):
        # return atan2(self.vy, self.vx)
        return fmod(self.direction- self.theta, TAU)

    @property
    def tangent(self):
        return ((atan2(-self.vx,self.vy) + (PI if self.vy > 0 else 0)) if self.vy != 0 else HALF_PI) % TAU
    

    @property
    def mag(self):
        return mag(self.vx, self.vy)

    @property
    def lift_force(self):
        # angle = (self.theta - self.direction) % TAU

        y = 0.7 * 1.225 * 0.75 / (2*mass)
        # normal lift
        mul = 50*y * self.mag**2 * cos(self.AoA) * sin(self.AoA)
        return [mul*cos(self.tangent), mul*sin(self.tangent)]


    def simulate(self):
        if not self.alive:
            return
        vx, vy = self.vx, self.vy

        # list of forces acting on player
        # drag
        # gravity
        # lift is related to the difference of movement angle and pointing angle
        # it points in the perpendicular to the direction of motion, favoring the side that the pointing angle is on.
        L = self.lift_force
        ax = L[0]
        # ax += influence * cos(self.theta)
        ay = L[1] - g * mass
        # ay += -influence * sin(self.theta) -0 g * mass
        _mag = self.mag
        # f = drag_narrow + (drag_flat-drag_narrow) * abs(sin(self.AoA))
        f = drag_narrow #+ (drag_flat-drag_narrow) * abs(sin(self.AoA))
        ax -= f * _mag * vx
        ay -= f * _mag * vy
        vx += ax / mass
        vy += ay / mass
        self.vx, self.vy = vx, vy
        self.x += self.vx * timedelta
        self.y -= self.vy * timedelta
        self.time += timedelta

    def reset(self):
        self.x, self.y, self.vx, self.vy = 0, 0, 0, 0
        self.theta = 0
        self.alive = True
        self.time = 0
        self.fitness = None

    def update(self):
        self.theta = self.brain.evaluate()

    def out_of_bounds(self):
        return self.x < -100 or self.y > self.target[1]

    def copy(self):
        return Player(self.target, params=self.brain.params[:])

    def transform_pos(self):
        """returns the coordinates to draw self to the screen"""
        return transform_pos((self.x, self.y))


def transform_pos(pos=(0,0)):
    """returns the screen coordinates to draw pos"""
    return vadd(OFFSET, (int(pos[0] / scale), int(pos[1] / scale)))


def reverse_transform_position(pos):
    """returns the approximate real coordinates that correspond to screen coordinates"""
    return vsub([e*scale for e in pos], OFFSET)


def redraw_screen(screen, DEST, color1, color2, color3, color4):
    screen.fill(color1)

    # pygame.draw.rect(screen, color2, pygame.Rect(vadd(OFFSET, (0, 0)), list(int(e / scale) for e in DEST)))
    screen.blit(color2, vadd(OFFSET, (0,0)))
    pygame.draw.circle(screen, color3, vadd(OFFSET, (0, 0)), 5)
    pygame.draw.circle(screen, color4, vadd(OFFSET, list(int(e / scale) for e in DEST)), 5)

def fitness_formula(pos, target, time):
    return FITNESS_HYPERPARAMETER_HEIGHT * exp(-(mag(*vsub(target, pos)) / FITNESS_HYPERPARAMETER_WIDTH) ** 2) - 2*time


def prepare_bg(target, SIZE, fitness_formula):
    surface = pygame.Surface((SIZE)).convert_alpha()
    for y in range(0, SIZE[1], 10):
        for x in range(0, SIZE[0], 10):
            pos = reverse_transform_position((x, y))
            v = fitness_formula(pos, target, 0)
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
        bg = prepare_bg(DEST, vsub(transform_pos(DEST), OFFSET), fitness_formula)

    if not HEADLESS:
        font = pygame.font.SysFont(pygame.font.get_default_font(), 22)
        resources.font = font

        redraw_screen(screen, DEST, WHITE, bg, GREEN, RED)

    flyers = construct_players(DEST, read_file)
    print(flyers[0].theta)
    print(flyers[0].brain.params)

    userPlayer = Player(DEST)
    best_fitness = float("inf")

    for i in range(BATCHES):
        frame = 0
        if frame % FRAMESKIP == 0 and not HEADLESS:
            headless_flag = False
        else:
            headless_flag = True
        print(f"batch {i} of {BATCHES} = {round(100*i/BATCHES,2)}% done")
        print(f"best fitness was {best_fitness}")
        halted = False

        if not headless_flag:
            redraw_screen(screen, DEST, WHITE, bg, GREEN, RED)

        alive_flyers_count = len(flyers)
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
                            for flyer in flyers:
                                flyer.reset()
                            userPlayer.reset()
                        elif e.key == K_x:
                            # exit and save
                            if should_write_training_data:
                                with open("save_data.json", "w") as fd:
                                    training_data = {"training_data": [flyer.brain.params for flyer in flyers]}
                                    json.dump(training_data, fd, indent=4)
                            pygame.quit()
                            return

            if not headless_flag and not args.show_trails:
                screen.fill(WHITE)
                redraw_screen(screen, DEST, WHITE, bg, GREEN, RED)

            render_text(f"fittest flyers momentum = {mass * mag(flyers[0].vx, flyers[0].vy)}")
            # render_text(f"user flyers momentum = {mass * mag(userPlayer.vx, userPlayer.vy)}")
            # render_text(f"user flyers speed = {1 * mag(userPlayer.vx, userPlayer.vy)}")
            # render_text(f"user flyers direction = {userPlayer.direction}")
            # render_text(f"user flyers theta = {userPlayer.theta}")
            # render_text(f"user flyers AoA = {userPlayer.AoA}")
            # render_text(f"user flyers velocity = {[userPlayer.vx, userPlayer.vy]}")

            for flyer in reversed(flyers):
                flyer.simulate()
                flyer.update()
                if flyer.alive and flyer.out_of_bounds():
                    flyer.alive = False
                    flyer.fitness = fitness_formula((flyer.x, flyer.y), flyer.target, flyer.time)
                    # print(flyer.fitness)
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, flyer.transform_pos(), 3)
                        render_text(flyer.fitness)
                    alive_flyers_count -= 1
                    continue
                if flyer.alive and flyer.time > TTL:
                    flyer.alive = False
                    flyer.fitness = fitness_formula((flyer.x, flyer.y), flyer.target, flyer.time)
                    # print(flyer.fitness)
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, flyer.transform_pos(), 3)
                        render_text(flyer.fitness)
                    alive_flyers_count -= 1
                    continue
                if not headless_flag:
                    pygame.draw.circle(screen, BLACK, flyer.transform_pos(), 1)

            userPlayer.simulate()
            if not headless_flag:
                userPlayer.theta = -atan2(
                    *vsub([x * scale for x in vsub(pygame.mouse.get_pos()[::-1], OFFSET)], [userPlayer.y, userPlayer.x])
                )
            else:
                # userqPlayer.theta = PI / 2
                pass
            # print(userPlayer.theta)
            if userPlayer.alive and userPlayer.out_of_bounds():
                userPlayer.alive = False
            if userPlayer.alive and userPlayer.time > TTL:
                userPlayer.alive = False
            if not headless_flag:
                pygame.draw.circle(screen, GREEN, userPlayer.transform_pos(), 1)
            if not headless_flag and should_draw_vectors:
                pygame.draw.line(
                    screen,
                    RED,
                    userPlayer.transform_pos(),
                    vadd(
                        [3 * influence * cos(userPlayer.theta) / mass, -3 * influence * sin(userPlayer.theta) / mass],
                        userPlayer.transform_pos(),
                    ),
                    1,
                )
                pygame.draw.line(
                    screen,
                    BLACK,
                    userPlayer.transform_pos(),
                    vadd(
                        [userPlayer.lift_force[0], -userPlayer.lift_force[1]],
                        userPlayer.transform_pos(),
                    ),
                    1,
                )
                pygame.draw.line(
                    screen,
                    BLACK,
                    userPlayer.transform_pos(),
                    vadd(
                        [userPlayer.vx, -userPlayer.vy],
                        userPlayer.transform_pos(),
                    ),
                    1,
                )

            render_text(f"alive flyers = {alive_flyers_count}")

            if not headless_flag:
                screen.blit(WHITE_SURFACE, (0,0), (0,0, longest_width+500, offset))
                for text_surface, pos in text_to_render:
                    screen.blit(text_surface, pos)
            if alive_flyers_count == 0:
                halted = True

            if not headless_flag:
                pygame.display.flip()
            clear_text_buffer()
            frame += 1
        best_fitness = max(*[e.fitness for e in flyers])
        new_target = random(RANDOM_LOWER_BOUND, RANDOM_UPPER_BOUND), FLOOR
        print("new target = " + str(new_target))
        for flyer in flyers:
            flyer.target = new_target
        flyers = selection_crossover_and_breeding(flyers)
        for flyer in flyers:
            flyer.reset()
        DEST = new_target
        userPlayer.reset()
        print(id(flyers[0]))
    if should_write_training_data:
        with open(write_file, "w") as fd:
            training_data = {"training_data": [flyer.brain.params for flyer in flyers]}
            json.dump(training_data, fd, indent=4)
            # fd.write("{\n")
            # for flyer in flyers:
            #     fd.write("\t[" + ", ".join(str(e) for e in flyer.brain.params) + "],\n")
            # fd.write("}\n")


if __name__ == "__main__":
    main()