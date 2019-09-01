
#include <iostream>
#include <random>
#include <cmath>
// from math import pi as PI
// from math import cos, sin, atan2, atan, tanh, fmod, exp, isclose
#include <ctime>
#include <utility>

#define MASS 100
#define Y_CONST 0.7 * 1.225 * 0.75 / (2*MASS);

float* crossover(float* brain1, float* brain2){
    // bits are basically the binary choices for which parameter to take from which parent
    if hasattr(rand, "choices"):
        bits = rand.choices([0, 1], k=N_PARAMS)
    else:
        bits = [rand.choice([0, 1]) for _ in range(N_PARAMS)]
    new_brain = [[brain1[i], brain2[i]][bit] for i, bit in enumerate(bits)]
    return new_brain
}

void mutation(float* brain){
    for i, param in enumerate(brain):
        if random(0, 1) < MUTATION_CHANCE:
            // brain.brain.params[i] = random(-MUTATION_EFFECT, MUTATION_EFFECT) + param
            brain[i] = rand.gauss(0, 2 * MUTATION_EFFECT) + param;
}


float* generate_child(float** brains){
    brain1, brain2 = rand.sample(brains, 2)
    new_brain = crossover(brain1, brain2)
    mutation(new_brain)
    yield new_brain
    i += 1
}


float selection_crossover_and_breeding(fitnesses, brains):
	static float[4] PROPORTIONS = {0.25, 0.25, 0.6, 0.15};
    auto new_brains = new std::vector();
    // sort by fitness, lower is better
    zipped = list(zip(fitnesses, brains))
    zipped.sort(key=lambda e: e[0], reverse=True)
    brains = list(zip(*zipped))[1]
    // truncate X% worst players
    brains = brains[: int(PROPORTIONS[0]*N_PLAYERS)]
    // keep the Y% greatest brains
    new_brains.extend(brains[: int(PROPORTIONS[1]*N_PLAYERS)])

    // breed brains to fill Z% of the new population
    for new_brain in generate_children(brains, n=int(PROPORTIONS[2]*N_PLAYERS)):
        new_brains.append(new_brain)
    // fill the rest with new randoms
    while len(new_brains) < N_PLAYERS:
        new_brains.append(construct_brain())
    assert len(new_brains) == N_PLAYERS, (len(new_brains), N_PLAYERS)
    return new_brains

float* construct_brain(){
    float* ret = new float(N_PARAMS);
    for (int i = 0; i < N_PARAMS; ++i)
    {
    	ret[i] = random(PARAM_LOWER_BOUND, PARAM_UPPER_BOUND);
    }
    return ret;
}

float evaluate(variables, params){
    variables[0], // x
    variables[1], // y
    pmag(variables), // mag
    direction(variables), // dir
    variables[4], // theta
    mag(variables[5]-variables[0], variables[6]-variables[1]), // distance to target
    atan2(variables[6]-variables[1], variables[5]-variables[0]), // angle to target
    variables[5]-variables[0], // horizontal distance to target
    variables[6]-variables[1], // vertical distance to target
    variables[7] // time

    rm = remap(variables)
    return fmod(
        params[-1] + sum(p * s for p, s in zip(params, remap(variables))),
        TAU
    )
}

bool out_of_bounds(float* player){
    return player[0] < -100 or player[1] > player[6];
}


float direction(player){
    return player[2] != 0 ? atan2(player[3], player[2]) : (PI+HALF_PI);
}

float AoA(player){
    // return atan2(player[3], player[2])
    return fmod(direction(player) - player[4], TAU);
}

float tangent(player){
    return ((atan2(-player[2],player[3]) + (PI if player[3] > 0 else 0)) if player[3] != 0 else HALF_PI) % TAU;
}


float pmag(player){
    return mag(player[2], player[3]);
}


std::pair<float, float> lift_force(player){
    // angle = (self.theta - self.direction) % TAU
    // normal lift
    _AoA = AoA(player)
    _tan = tangent(player)
    mul = 50*Y_CONST * pmag(player)**2 * cos(_AoA) * sin(_AoA)
    return [mul*cos(_tan), mul*sin(_tan)];
}

float simulate(player):
    L = lift_force(player)
    _mag = mag(player[2], player[3])
    // f = drag_narrow + (drag_flat-drag_narrow) * abs(sin(self.AoA))
    f = drag_narrow
    player[2] += (L[0] - f * _mag * player[2])/mass
    player[3] += (L[1] - g * mass - f * _mag * player[3])/mass
    player[0] += player[2] * timedelta
    player[1] -= player[3] * timedelta
    player[7] += timedelta

float update(player, brain):
    player[4] = evaluate(player, brain)

float construct_player(target):
    return [
        0, // x
        0, // y
        0, // vx
        0, // vy
        0, // theta
        target[0], // target x
        target[1], // target y
        0, // time
        True, // alive
        0 // fitness
    ]


float reset(player):
    player[0:5] = [0,0,0,0,0]
    player[7] = 0
    player[-2] = True
    player[-1] = 0


float construct_player_and_brain(target, params=None):
    player = construct_player(target)
    if params is None:
        brain = construct_brain()
    else:
        brain = params[:]
    update(player, brain) 
    return player, brain

float construct_players_and_brains(DEST, filename=None):
    if should_read_training_data:
        with open(filename, "r") as fd:
            data = json.load(fd)

    players = []
    brains = []
    if should_read_training_data:
        for i in range(len(data["training_data"])):
            players.append(construct_player(DEST))
            brains.append(data["training_data"][i][:])
            update(players[-1], brains[-1])
    else:
        for i in range(N_PLAYERS):
            players.append(construct_player(DEST))
            brains.append(construct_brain())
            update(players[-1], brains[-1]) 
    return players, brains

float transform_pos(x=0, y=0):
    """returns the screen coordinates to draw pos"""
    return vadd(OFFSET, (int(x / scale), int(y / scale)))


float reverse_transform_position(x, y):
    """returns the approximate real coordinates that correspond to screen coordinates"""
    return vsub([e*scale for e in [x,y]], OFFSET)


float redraw_screen(screen, DEST, color1, color2, color3, color4):
    screen.fill(color1)

    // pygame.draw.rect(screen, color2, pygame.Rect(vadd(OFFSET, (0, 0)), list(int(e / scale) for e in DEST)))
    screen.blit(color2, vadd(OFFSET, (0,0)))
    pygame.draw.circle(screen, color3, vadd(OFFSET, (0, 0)), 5)
    pygame.draw.circle(screen, color4, vadd(OFFSET, list(int(e / scale) for e in DEST)), 5)


float player_fitness_formula(player):
    return fitness_formula(player[0], player[1], player[5], player[6], player[7])

float fitness_formula(x, y, tx, ty, time):
    return FITNESS_HYPERPARAMETER_HEIGHT * exp(-(mag(tx-x, ty-y) / FITNESS_HYPERPARAMETER_WIDTH) ** 2) - 2*time



int main(int argc, char const *argv[]) {
	// constants
	bool should_write_training_data = false;
	bool should_read_training_data = false;
	int BATCHES = 100;
	int N_PLAYERS = 128;


	float HALF_PI = PI / 2;
	float TAU = PI * 2;

	float g = 9.80665;
	float drag_flat = 0.07;
	float drag_narrow = 0.038;

	float MUTATION_EFFECT = 0.20;
	float MUTATION_CHANCE = 0.20;
	float timedelta = 0.1;

	float N_PARAMS = 11;
	float TTL = 1000;
	float RANDOM_LOWER_BOUND = 15000;
	float RANDOM_UPPER_BOUND = 15000;
	float PARAM_LOWER_BOUND = -1;
	float PARAM_UPPER_BOUND = 1;
	float RANDOM_INITIAL = 15000;
	float FITNESS_HYPERPARAMETER_WIDTH=10000;
	float FITNESS_HYPERPARAMETER_HEIGHT=30000;


	assert fitness_formula(0,0,0,0,0) == 30000
	

    FLOOR = 10000
    // DEST = RANDOM_INITIAL, FLOOR


    players, brains = construct_players_and_brains(DEST);

    for i in range(BATCHES):
        t1 = time.perf_counter()
        frame = 0
        halted = False

        alive_players_count = len(players)
        while not halted:
            for player, brain in zip(reversed(players), reversed(brains)):
                if not player[-2]:
                    continue
                simulate(player)
                update(player, brain)
                if player[-2] and out_of_bounds(player):
                    player[-2] = False // assign alive status
                    player[-1] = player_fitness_formula(player) // assign fitness
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, transform_pos(player[0], player[1]), 3)
                        render_text(player[-1])
                    alive_players_count -= 1
                    continue
                if player[-2] and player[7] > TTL:
                    player[-2] = False // assign alive status
                    player[-1] = player_fitness_formula(player) // assign fitness
                    if not headless_flag:
                        pygame.draw.circle(screen, RED, transform_pos(player[0], player[1]), 3)
                        render_text(player[-1])
                    alive_players_count -= 1
                    continue
                if not headless_flag:
                    pygame.draw.circle(screen, BLACK, transform_pos(player[0], player[1]), 1)

            if alive_players_count == 0:
                halted = True

            frame++;
        t2 = time.perf_counter()
        best_fitness = max(*[e[-1] for e in players])
        new_target = random(RANDOM_LOWER_BOUND, RANDOM_UPPER_BOUND), FLOOR
        for player in players:
            player[5:7] = new_target
        brains = selection_crossover_and_breeding([p[-1] for p in players], brains)
        // assert len(set(id(e) for e in brains)) == len(brains)
        players = players[:len(brains)]
        for player in players:
            reset(player)
        DEST = new_target
        reset(userPlayer)
        t3 = time.perf_counter()
    return 0;
}
