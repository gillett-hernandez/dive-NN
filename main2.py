HEADLESS = True
if not HEADLESS:
    import pygame
    from pygame.locals import *

# import numpy as np
from random import uniform as random
import random as rand
from math import pi as PI
from math import cos, sin, atan2, atan, tanh, fmod, exp
from math import hypot as mag
import json

# import tensorflow as tf


HALF_PI = PI / 2
TAU = PI * 2

N_PLAYERS = 512
g = 9.80665
f = 0.1
mass = 100
scale = 15
influence = 700
timedelta = 0.1
N_PARAMS = 11
MUTATION_EFFECT = 0.10
MUTATION_CHANCE = 0.20
TTL = 1000
BATCHES = 100
OFFSET = (50, 50)
should_write_training_data = True
should_read_training_data = True
RANDOM_LOWER_BOUND = 9000
RANDOM_UPPER_BOUND = 9000
PARAM_LOWER_BOUND = -3
PARAM_UPPER_BOUND = 3
RANDOM_INITIAL = 9000
# only show every FRAMESKIP frame
FRAMESKIP = 10


def crossover(flyer1, flyer2):
    # bits are basically the binary choices for which parameter to take from which parent
    bits = rand.choices([0, 1], k=N_PARAMS)
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


def selection_crossover_and_breeding(flyers):
    new_flyers = []
    # sort by fitness, lower is better
    flyers.sort(key=lambda e: e.fitness, reverse=True)
    # truncate 50% worst flyers
    flyers = flyers[: int(len(flyers) / 2)]
    # keep the greatest flyers
    new_flyers.extend(flyers)
    # shuffle flyers so we can split them into random batches
    rand.shuffle(flyers)
    # zip through and breed flyers
    for i, (flyer1, flyer2) in enumerate(zip(flyers[: int(len(flyers) / 2)], flyers[int(len(flyers) / 2) :])):
        for k in range(2):
            # random crossover
            new_flyer = crossover(flyer1, flyer2)
            # random mutations
            mutation(new_flyer)
            new_flyers.append(new_flyer)
    assert len(new_flyers) == N_PLAYERS, (len(new_flyers), N_PLAYERS)
    return new_flyers


def sign(x):
    return 1 if x >= 0 else -1


def vadd(l1, l2):
    return l1[0] + l2[0], l1[1] + l2[1]


def vsub(l1, l2):
    return l1[0] - l2[0], l1[1] - l2[1]


class Brain:
    __slots__ = ["flyer", "params"]

    def __init__(self, flyer):
        self.flyer = flyer
        self.params = [random(-PARAM_LOWER_BOUND, PARAM_UPPER_BOUND) for _ in range(N_PARAMS)]
        # self.bias = random(-2,2)

    def evaluate(self):
        S = fmod(
            self.params[-1]
            + sum(
                p * s
                for p, s in zip(
                    self.params,
                    [
                        self.flyer.x,
                        self.flyer.y,
                        mag(self.flyer.vx, self.flyer.vy),
                        atan2(self.flyer.vy, self.flyer.vx),
                        self.flyer.theta,
                        mag(*vsub(self.flyer.target, [self.flyer.x, self.flyer.y])),
                        atan2(*vsub(self.flyer.target, [self.flyer.x, self.flyer.y])),
                        (self.flyer.target[0] - self.flyer.x),
                        (self.flyer.target[1] - self.flyer.y),
                        self.flyer.time,
                    ],
                )
            ),
            TAU,
        )
        # S = sum(p*p2*s for p,s,p2 in zip(self.params, [self.flyer.x, self.flyer.y, mag(self.flyer.vx, self.flyer.vy), atan2(self.flyer.vy,self.flyer.vx), self.flyer.theta], self.params2))
        # print(S)
        return S


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
        self.theta = self.brain.evaluate()
        self.alive = True
        self.fitness = None

    def simulate(self):
        if not self.alive:
            return
        vx, vy = self.vx, self.vy
        ax, ay = 0, 0

        ax += influence * cos(self.theta)
        ay += -influence * sin(self.theta) - g * mass
        _mag = mag(vx, vy)
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
        self.alive = True
        self.time = 0
        self.fitness = None

    def update(self):
        self.theta = self.brain.evaluate()

    def out_of_bounds(self):
        return self.x < 0 or self.y > self.target[1]

    def copy(self):
        cp = Player(self.target, params=self.brain.params[:])
        return cp

    def transform_pos(self):
        """returns the coordinates to draw self to the screen"""
        return vadd(OFFSET, (int(self.x / scale), int(self.y / scale)))


def fitness_formula(flyer):
    return 1000 * exp(-(mag(*vsub(flyer.target, [flyer.x, flyer.y])) / 1000) ** 2) - flyer.time


def main():
    global HEADLESS
    if not HEADLESS:

        pygame.init()

    SIZE = WIDTH, HEIGHT = (1000, 1000)
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

    if not HEADLESS:
        font = pygame.font.SysFont(pygame.font.get_default_font(), 22)

    if should_read_training_data:
        with open("save_data.json", "r") as fd:
            data = json.load(fd)

    flyers = []
    for i in range(N_PLAYERS):
        flyers.append(Player(DEST))
        if should_read_training_data:
            flyers[-1].brain.params = data["training_data"][i]
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
            screen.fill(WHITE)

            pygame.draw.rect(screen, GREY, pygame.Rect(vadd(OFFSET, (0, 0)), list(int(e / scale) for e in DEST)))
            pygame.draw.circle(screen, GREEN, vadd(OFFSET, (0, 0)), 5)
            pygame.draw.circle(screen, RED, vadd(OFFSET, list(int(e / scale) for e in DEST)), 5)

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
                                canvas.fill(WHITE)
                            for flyer in flyers:
                                flyer.reset()

            # screen.fill(WHITE)

            for flyer in flyers:
                flyer.simulate()
                flyer.update()
                if flyer.alive and flyer.out_of_bounds():
                    flyer.alive = False
                    flyer.fitness = fitness_formula(flyer)
                    # print(flyer.fitness)
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, flyer.transform_pos(), 3)
                    alive_flyers_count -= 1
                    continue
                if flyer.alive and flyer.time > TTL:
                    flyer.alive = False
                    flyer.fitness = fitness_formula(flyer)
                    # print(flyer.fitness)
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, flyer.transform_pos(), 3)
                    alive_flyers_count -= 1
                    continue
                if not headless_flag:
                    pygame.draw.circle(screen, BLACK, flyer.transform_pos(), 1)

            userPlayer.simulate()
            if not headless_flag:
                userPlayer.theta = atan2(
                    *vsub([x * scale for x in vsub(pygame.mouse.get_pos()[::-1], OFFSET)], [userPlayer.y, userPlayer.x])
                )
            else:
                userPlayer.theta = PI / 2
            # print(userPlayer.theta)
            if userPlayer.alive and userPlayer.out_of_bounds():
                userPlayer.alive = False
            if userPlayer.alive and userPlayer.time > TTL:
                userPlayer.alive = False
            if not headless_flag:
                pygame.draw.circle(screen, GREEN, userPlayer.transform_pos(), 1)
            if not headless_flag:
                pygame.draw.line(
                    screen,
                    RED,
                    userPlayer.transform_pos(),
                    vadd(
                        [3 * influence * cos(userPlayer.theta) / mass, 3 * influence * sin(userPlayer.theta) / mass],
                        userPlayer.transform_pos(),
                    ),
                    1,
                )

            render_str = f"alive flyers = {alive_flyers_count}"
            if not headless_flag:
                screen.blit(WHITE_SURFACE, (WIDTH - 200, 10, *font.size(render_str)))
                screen.blit(font.render(render_str, False, BLACK), (WIDTH - 200, 10))
            if alive_flyers_count == 0:
                halted = True

            if not headless_flag:
                pygame.display.flip()
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
        with open("save_data.json", "w") as fd:
            training_data = {"training_data": [flyer.brain.params for flyer in flyers]}
            json.dump(training_data, fd, indent=4)
            # fd.write("{\n")
            # for flyer in flyers:
            #     fd.write("\t[" + ", ".join(str(e) for e in flyer.brain.params) + "],\n")
            # fd.write("}\n")


if __name__ == "__main__":
    main()
