#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--read-data", action="store_true")
parser.add_argument("--write-data", action="store_true")
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--batches", type=int, default=100)
parser.add_argument("--n-players", type=int, default=1024)


args = parser.parse_args()

HEADLESS = args.headless
FRAMESKIP = args.frameskip
should_write_training_data = args.write_data
should_read_training_data = args.read_data
BATCHES = args.batches
N_PLAYERS = args.n_players

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
import tensorflow as tf

tf.enable_eager_execution()

HALF_PI = PI / 2
TAU = PI * 2

g = 9.80665
g = 1
drag_flat = 0.10
drag_narrow = 0.05
mass = 80
scale = 15
influence = 900

MUTATION_EFFECT = 0.20
MUTATION_CHANCE = 0.20
timedelta = 0.1

N_PARAMS = 11
TTL = 1000
OFFSET = (50, 50)
RANDOM_LOWER_BOUND = 9000
RANDOM_UPPER_BOUND = 9000
PARAM_LOWER_BOUND = -3
PARAM_UPPER_BOUND = 3
RANDOM_INITIAL = 9000
FITNESS_HYPERPARAMETER_WIDTH=10000
FITNESS_HYPERPARAMETER_HEIGHT=30000


def crossover(player1, player2):
    # bits are basically the binary choices for which parameter to take from which parent
    if hasattr(rand, "choices"):
        bits = rand.choices([0, 1], k=N_PARAMS)
    else:
        bits = [rand.choice([0, 1]) for _ in range(N_PARAMS)]
    params1 = player1.brain.params
    params2 = player2.brain.params
    new_player = Player(player1.target)
    new_player.brain.params = [[params1[i], params2[i]][bit] for i, bit in enumerate(bits)]
    return new_player


def mutation(player):
    for i, param in enumerate(player.brain.params):
        if random(0, 1) < MUTATION_CHANCE:
            # print("mutation occurred!")
            # player.brain.params[i] = random(-MUTATION_EFFECT, MUTATION_EFFECT) + param
            player.brain.params[i] = rand.gauss(0, 2 * MUTATION_EFFECT) + param


def generate_children(players):
    while True:
        player1, player2 = rand.sample(players, 2)
        new_player = crossover(player1, player2)
        mutation(new_player)
        yield new_player


def selection_crossover_and_breeding(players):
    new_players = []
    # sort by fitness, lower is better
    players.sort(key=lambda e: e.fitness, reverse=True)
    # truncate 50% worst players
    players = players[: int(len(players) / 2)]
    # keep the greatest players
    new_players.extend(players)
    # shuffle players so we can split them into random batches
    rand.shuffle(players)
    # zip through and breed players
    for i, new_player in zip(range(int(len(players)//4)), generate_children(players)):
        new_players.append(new_player)
    while len(new_players) < N_PLAYERS:
        new_players.append(Player(new_players[-1].target))
    assert len(new_players) == N_PLAYERS, (len(new_players), N_PLAYERS)
    return new_players


def sign(x):
    return 1 if x >= 0 else -1


def vadd(l1, l2):
    return l1[0] + l2[0], l1[1] + l2[1]


def vsub(l1, l2):
    return l1[0] - l2[0], l1[1] - l2[1]

def dot(v1, v2):
    return sum([e1*e2 for e1,e2 in zip(v1,v2)])


class Brain:
    __slots__ = ["player", "params"]

    def __init__(self, player):
        self.player = player
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
                        self.player.x,
                        self.player.y,
                        self.player.mag,
                        self.player.direction,
                        self.player.theta,
                        mag(*vsub(self.player.target, [self.player.x, self.player.y])),
                        atan2(*vsub(self.player.target, [self.player.x, self.player.y])),
                        (self.player.target[0] - self.player.x),
                        (self.player.target[1] - self.player.y),
                        self.player.time,
                    ],
                )
            ),
            TAU,
        )
        # S = sum(p*p2*s for p,s,p2 in zip(self.params, [self.player.x, self.player.y, mag(self.player.vx, self.player.vy), atan2(self.player.vy,self.player.vx), self.player.theta], self.params2))
        # print(S)
        return S


class Player:
    __slots__ = ["x", "y", "vx", "vy", "theta", "brain", "alive", "target", "time", "fitness"]

    def __init__(self, target, params=None):
        self.x = 0
        self.y = 0
        self.vx = 20
        self.vy = 0
        # self.intent = 0
        self.theta = 0
        self.time = 0
        self.brain = Brain(self)
        if params is not None:
            self.brain.params = params
        self.target = target
        self.theta = self.brain.evaluate(self)
        self.alive = True
        self.fitness = None

    @property
    def direction(self):
        return atan2(self.vy, self.vx)

    @property
    def AoA(self):
        # return atan2(self.vy, self.vx)
        return self.theta - self.direction

    @property
    def mag(self):
        return mag(self.vx, self.vy)

    @property
    def lift_force(self):
        # angle = (self.theta - self.direction) % TAU

        y = 0.7 * 1.225 * 0.75 / (2*mass)
        # CI = TAU*self.AoA
        CI = 1
        # normal lift
        mul = 30*y * CI * self.mag**2 * cos(self.AoA) * sin(self.AoA)
        return [mul * x for x in [cos(HALF_PI + self.direction), sin(HALF_PI + self.direction)]]

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
        # f = drag_narrow if abs(sin(self.AoA)) < 0.3 else drag_flat
        f = drag_flat
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
        self.fitness = None

    def update(self):
        self.theta = self.brain.evaluate(self)

    def out_of_bounds(self):
        return self.x < -100 or self.y > self.target[1]

    def copy(self):
        return Player(self.target, params=self.brain.params[:])

    def transform_pos(self):
        """returns the coordinates to draw self to the screen"""
        return vadd(OFFSET, (int(self.x / scale), int(self.y / scale)))



def redraw_screen(screen, DEST, color1, color2, color3, color4):
    screen.fill(color1)

    pygame.draw.rect(screen, color2, pygame.Rect(vadd(OFFSET, (0, 0)), list(int(e / scale) for e in DEST)))
    pygame.draw.circle(screen, color3, vadd(OFFSET, (0, 0)), 5)
    pygame.draw.circle(screen, color4, vadd(OFFSET, list(int(e / scale) for e in DEST)), 5)

def fitness_formula(player):
    return FITNESS_HYPERPARAMETER_HEIGHT * exp(-(mag(*vsub(player.target, [player.x, player.y])) / FITNESS_HYPERPARAMETER_WIDTH) ** 2) - player.time



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

    players = []
    for i in range(N_PLAYERS):
        players.append(Player(DEST))
        players[-1].theta = random(0,TAU)
        if should_read_training_data:
            if "version" not in data or data["version"] < 1:
                length_difference = len(players[-1].brain.params) - len(data["training_data"][i])
                players[-1].brain.params = data["training_data"][i] + [0] * length_difference
            else:
                players[-1].brain.params = data["training_data"][i]
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

        if not HEADLESS:
            redraw_screen(screen, DEST, WHITE, GREY, GREEN, RED)


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
                            if not HEADLESS:
                                redraw_screen(screen, DEST, WHITE, GREY, GREEN, RED)
                            userPlayer.reset()
                            for player in players:
                                player.reset()

            # screen.fill(WHITE)

            for player in players:
                player.simulate()
                player.update()
                if player.alive and player.out_of_bounds():
                    player.alive = False
                    player.fitness = fitness_formula(player)
                    # print(player.fitness)
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, player.transform_pos(), 3)
                    alive_players_count -= 1
                    continue
                if player.alive and player.time > TTL:
                    player.alive = False
                    player.fitness = fitness_formula(player)
                    # print(player.fitness)
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, player.transform_pos(), 3)
                    alive_players_count -= 1
                    continue
                if not headless_flag:
                    pygame.draw.circle(screen, BLACK, player.transform_pos(), 1)

            userPlayer.simulate()
            print(userPlayer.x, userPlayer.y,  userPlayer.vx, userPlayer.vy, userPlayer.theta, userPlayer.direction, userPlayer.AoA)

            if not headless_flag:
                userPlayer.theta = atan2(
                    # *vsub([x * scale for x in vsub(pygame.mouse.get_pos()[::-1], OFFSET)], [userPlayer.y, userPlayer.x])
                    *[x * scale for x in vsub(pygame.mouse.get_pos()[::-1], OFFSET)]
                )
            else:
                userPlayer.theta = PI / 4
            # print(userPlayer.theta)
            if userPlayer.alive and userPlayer.out_of_bounds():
                userPlayer.alive = False
            if userPlayer.alive and userPlayer.time > TTL:
                userPlayer.alive = False
            if not headless_flag:
                pygame.draw.circle(screen, GREEN, userPlayer.transform_pos(), 1)
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
                pygame.draw.line(
                    screen,
                    GREEN,
                    userPlayer.transform_pos(),
                    vadd(
                        [3 * influence * cos(userPlayer.direction) / mass, -3 * influence * sin(userPlayer.direction) / mass],
                        userPlayer.transform_pos(),
                    ),
                    1,
                )
                pygame.draw.line(
                    screen,
                    BLACK,
                    userPlayer.transform_pos(),
                    vadd(
                        userPlayer.lift_force,
                        userPlayer.transform_pos(),
                    ),
                    1,
                )

            render_str = f"alive players = {alive_players_count}"
            if not headless_flag:
                screen.blit(WHITE_SURFACE, (WIDTH - 200, 10, *font.size(render_str)))
                screen.blit(font.render(render_str, False, BLACK), (WIDTH - 200, 10))
            if alive_players_count == 0:
                halted = True

            if not headless_flag:
                pygame.display.flip()
            frame += 1
        best_fitness = max(*[e.fitness for e in players])
        new_target = random(RANDOM_LOWER_BOUND, RANDOM_UPPER_BOUND), FLOOR
        print("new target = " + str(new_target))
        for player in players:
            player.target = new_target
        players = selection_crossover_and_breeding(players)
        for player in players:
            player.reset()
        DEST = new_target
        userPlayer.reset()
        userPlayer.x, userPlayer.y = 1000,1000
        print(id(players[0]))
    if should_write_training_data:
        with open("save_data.json", "w") as fd:
            training_data = {"training_data": [player.brain.params for player in players]}
            json.dump(training_data, fd, indent=4)
            # fd.write("{\n")
            # for player in players:
            #     fd.write("\t[" + ", ".join(str(e) for e in player.brain.params) + "],\n")
            # fd.write("}\n")


if __name__ == "__main__":
    main()
