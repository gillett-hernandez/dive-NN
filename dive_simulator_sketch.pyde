import math

N_PLAYERS = 1000
players = []

class Player:
    __slots__ = ["x", "y", "vx", "vy", "theta", "intent"]
    def __init__(self):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.theta = 0
        self.intent = 0

def setup():
    size(1000,1000)
    for i in range(N_PLAYERS):
        players.append(Player())
        player = players[-1]
        player.theta = random(PI+HALF_PI, TAU)

g = 9.80665
f = 0.1
m = 100
_scale = 100
influence = 10

def sign(x):
    return 1 if x >= 0 else -1

def simulate(player):
    vx, vy = player.vx, player.vy
    
    vx += influence*cos(player.theta)
    vy += influence*sin(player.theta) - g
    _mag = mag(vx, vy)
    vx -= f*vx
    #print(f*vx, f*vy)
    vy -= f*vy
    player.vx, player.vy = vx, vy
    player.x += player.vx
    player.y -= player.vy

def update(player):
    # player.intent += (player.intent+1)*random(-2,2)
    # player.theta = PI+HALF_PI+(HALF_PI + atan(player.intent))/2
    pass

def draw():
    background(255,255,255)
    stroke(0,0,0)
    fill(0,0,0)
    #print(players[0].x, players[0].y, players[0].vx, players[0].vy, cos(players[0].theta), sin(players[0].theta))
    for player in players:
        ellipse(100+player.x/_scale, 100+player.y/_scale, 8, 8)
        simulate(player)
        update(player)
