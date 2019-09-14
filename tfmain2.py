#!/usr/bin/env python3

import argparse
from random import uniform as random
import random as rand
from math import pi as PI
from math import cos, sin, atan2, atan, tanh, fmod, exp, isclose
import time
import json
from itertools import takewhile

import numpy as np
import tensorflow as tf

from namespace import Namespace


mag = lambda a, b: tf.math.sqrt(a ** 2 + b ** 2)


should_write_training_data = False
should_read_training_data = False
BATCHES = 100
N_PLAYERS = 8
debug = False

ORDER = "C"


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
FITNESS_HYPERPARAMETER_WIDTH = 10000
FITNESS_HYPERPARAMETER_HEIGHT = 30000


trues = tf.tile([True], [N_PLAYERS])
falses = tf.tile([False], [N_PLAYERS])
zeroes = tf.fill([N_PLAYERS], 0.0)
ones = tf.fill([N_PLAYERS], 1.0)
PI_AND_A_HALF = tf.tile([PI + HALF_PI], [N_PLAYERS])
tensor_PI = tf.tile([PI], [N_PLAYERS])
tensor_HALF_PI = tf.tile([PI], [N_PLAYERS])


BRAIN = None


def construct_brain():
    global BRAIN
    if BRAIN is None:
        # input layer shape is 10 because that's how many parameters batch_remap outputs.
        # BRAIN = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])
        input_dim = 10
        output_dim = 1
        timesteps = 4
        cells = [
            tf.keras.layers.LSTMCell(output_dim),
            tf.keras.layers.LSTMCell(output_dim),
            tf.keras.layers.LSTMCell(output_dim),
        ]

        inputs = tf.keras.Input((timesteps, input_dim))
        x = tf.keras.layers.RNN(cells)
        breakpoint()
        BRAIN = x
    #     return tf.constant([random(PARAM_LOWER_BOUND, PARAM_UPPER_BOUND) for _ in range(N_PARAMS)], dtype=tf.dtypes.float32)
    return BRAIN


def pmag(players):
    return mag(players[2], players[3])


def lift_force(players):
    # angle = (self.theta - self.direction) % TAU

    y = 0.7 * 1.225 * 0.75 / (2 * mass)
    # normal lift
    _AoA = AoA(players)
    _tan = tangent(players)
    mul = 50 * y * pmag(players) ** 2 * tf.cos(_AoA) * tf.sin(_AoA)
    return [mul * tf.cos(_tan), mul * tf.sin(_tan)]


def reset(DEST):
    return construct_players(DEST)


def construct_players(DEST, number=N_PLAYERS, **kwargs):
    DEST = [float(e) for e in DEST]
    if kwargs.get("random", False):
        return tf.stack(
            [
                tf.tile([0.0], [number]),
                tf.tile([0.0], [number]),
                tf.tile([0.0], [number]),
                tf.tile([0.0], [number]),
                tf.random.uniform(shape=(number,)),
                tf.tile([DEST[0]], [number]),
                tf.tile([DEST[1]], [number]),
                tf.tile([0.0], [number]),
                tf.tile([1.0], [number]),
                tf.tile([0.0], [number]),
            ]
        )


# def construct_players(DEST, **kwargs):
# #     players = []
# #     for i in range(N_PLAYERS):
# #         players.append(construct_player())
# #     return tf.transpose(tf.stack(players))
#     if kwargs.get("random", False):
#         print(tf.random.uniform(shape=(1,)))
#         return tf.transpose([construct_player(DEST, theta=tf.random.uniform(shape=(1,)))])
#     else:
#         return tf.transpose(tf.reshape(tf.tile(construct_player(DEST, **kwargs), [N_PLAYERS]), (N_PLAYERS, N_PARAMS-1)))


def construct_players_and_brain(DEST, random=False, filename=None, number=N_PLAYERS):
    players = construct_players(DEST, random=random, number=number)
    print(players.shape)

    brain = construct_brain()

    #     for i in range(N_PLAYERS):
    #         brains.append(construct_brain())
    #     tf.transpose(tf.stack(brains))
    return players, brain


def transform_pos(x=0, y=0):
    """returns the screen coordinates to draw pos"""
    return vadd(OFFSET, (int(x / scale), int(y / scale)))


def reverse_transform_position(x, y):
    """returns the approximate real coordinates that correspond to screen coordinates"""
    return vsub([e * scale for e in [x, y]], OFFSET)


def simulate_and_update(players, brain):
    L = lift_force(players)
    _mag = mag(players[2], players[3])
    f = drag_narrow
    l2 = players[2] + (L[0] - f * _mag * players[2]) / mass
    l3 = players[3] + (L[1] - g * mass - f * _mag * players[3]) / mass
    l0 = players[0] + l2 * timedelta
    l1 = players[1] - l3 * timedelta
    l7 = players[7] + timedelta
    l4 = brain.predict_on_batch(tf.transpose(batch_remap(players)))
    return tf.stack([l0, l1, l2, l3, l4[:, 0], players[5], players[6], l7, players[8], players[9]])


def update(players, brain):

    l4 = brain.predict_on_batch(tf.transpose(batch_remap(players)))
    print(l4, dir(l4), type(l4))
    return tf.stack(
        [
            players[0],
            players[1],
            players[2],
            players[3],
            l4[:, 0],
            players[5],
            players[6],
            players[7],
            players[8],
            players[9],
        ]
    )


def direction(players):
    a = tf.atan2(players[3], players[2])

    b = tf.where(players[2] != 0, zeroes, PI_AND_A_HALF)

    return a + b


def AoA(players):
    return tf.math.floormod(direction(players) - players[4], TAU)


def tangent(players):

    return tf.math.floormod(
        tf.where(
            players[3] != 0,
            tf.atan2(-players[2], players[3]) + tf.where(players[3] > 0, tensor_PI, zeroes),
            tensor_HALF_PI,
        ),
        TAU,
    )


def batch_remap(variables):
    ret = tf.stack(
        [
            variables[0],  # x
            variables[1],  # y
            pmag(variables),  # mag
            direction(variables),  # dir
            variables[4],  # theta
            mag(variables[5] - variables[0], variables[6] - variables[1]),
            tf.atan2(variables[5] - variables[0], variables[6] - variables[1]),
            variables[5] - variables[0],  # dx
            variables[6] - variables[1],  # dy
            variables[7],
        ]  # time
    )
    # print(ret, type(ret), dir(ret))
    return ret


def out_of_bounds(players):
    a = tf.where(players[0] < -100, trues, falses)
    b = tf.where(players[1] > players[6], trues, falses)
    return tf.logical_or(a, b)


def player_fitness_formula(players):
    return fitness_formula(players[0], players[1], players[5], players[6], players[7])


def fitness_formula(x, y, tx, ty, time):
    #     print("x", x)
    #     print("y", y)
    #     print("tx", tx)
    #     print("ty", ty)
    #     print("time", time)
    return (
        FITNESS_HYPERPARAMETER_HEIGHT
        * tf.exp(-(mag(tx - x, ty - y) / FITNESS_HYPERPARAMETER_WIDTH) ** 2)
        - 2 * time
    )


def update_death_and_fitness(players):
    dead = tf.logical_or(out_of_bounds(players), players[7] > TTL)
    pff = player_fitness_formula(players)
    #     print(pff)
    p_2 = tf.where(dead, zeroes, ones)
    p_1 = tf.where(dead, pff, zeroes)
    return tf.stack(
        [
            players[0],
            players[1],
            players[2],
            players[3],
            players[4],
            players[5],
            players[6],
            players[7],
            p_2,
            p_1,
        ]
    )


def main(read_file="savedata.json", write_file="savedata.json"):

    FLOOR = 10000
    DEST = RANDOM_INITIAL, FLOOR

    best_fitness = float("inf")
    brain = construct_brain()
    if True:
        for i in range(BATCHES):
            #             with tf.GradientTape() as grad:
            if True:
                players = construct_players(DEST, random=True)
                #                 grad.watch(players)
                players = update(players, brain)
                print(f"starting batch {i} of {BATCHES}")
                t1 = time.perf_counter()
                frame = 0
                halted = False

                alive_players_count = [N_PLAYERS]

                while not halted:
                    if frame % 10 == 0:
                        print(".", end="")
                    if frame % 100 == 1:
                        print(alive_players_count.numpy())
                    #                     print("p1", players)
                    players = simulate_and_update(players, brain)
                    #                     print("p2", players)
                    players = update_death_and_fitness(players)
                    #                     print("p3", players)
                    #                     print("p4", player_fitness_formula(players))
                    alive_players_count = tf.math.count_nonzero(players[-2])
                    if alive_players_count == 0:
                        halted = True

                    frame += 1
                t2 = time.perf_counter()

                best_fitness = max(*[e[-1] for e in players])
                print(f"best fitness was {best_fitness}")
            fitnesses = players[-1]
            print(fitnesses.shape)
            print(dir(grad))
            print(grad.gradient(fitnesses, brain.trainable_weights))

            return
            new_target = random(RANDOM_LOWER_BOUND, RANDOM_UPPER_BOUND), FLOOR
            print(f"batch {i} of {BATCHES} = {round(100*i/BATCHES,2)}% done")
            print("time for batch: {}".format(t2 - t1))
            # assert len(set(id(e) for e in brains)) == len(brains)
            #         players = players[:len(brains)]
            reset(DEST)
            DEST = new_target
            t3 = time.perf_counter()
            print("time for resetting: {}".format(t3 - t2))
            print("\n")


def test():
    test_players, brain = construct_players_and_brain((15000, 10000), random=True, number=8)
    print(test_players.shape)
    out_of_bounds(test_players)
    assert update(test_players, brain).shape == (10, 8)
    assert simulate_and_update(test_players, brain).shape == (10, 8)


if __name__ == "__main__":
    test()
    main()
