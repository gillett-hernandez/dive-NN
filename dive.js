var N_PLAYERS = 1000;
var players = [];

function Brain(player) {
	this.matrix = Matrix()
	this.evaluate = function(){
		return 
	}
	return this
}

function Player() {
	this.x = 0;
	this.y = 0;
	this.vx = 0;
	this.vy = 0;
	this.theta = 0;
	this.intent = 0;
	this.brain = new Brain(this)
	return this
}

var g = 9.80665;
var f = 0.1;
var m = 100;
var _scale = 100
var influence = 10

function sign(x){
	if (x >= 0) {
    return 1;
	} else{
		return -1;
	}
}

function simulate(player){
    var vx = player.vx;
		var vy = player.vy;
    
    vx += influence*cos(player.theta);
    vy += influence*sin(player.theta) - g;
    _mag = mag(vx, vy);
    vx -= f*vx;
    // print(f*vx, f*vy);
    vy -= f*vy;
    player.vx = vx;
	  player.vy = vy;
    player.x += player.vx;
    player.y -= player.vy;
}

function update(player){
	player.theta = player.brain.evaluate()
    // player.intent += (player.intent+1)*random(-6,6)
    // player.theta = PI+HALF_PI+(HALF_PI + atan(player.intent))/2
	// player.theta = randomGaussian(HALF_PI + PI, TAU)
}



function setup() {
	createCanvas(1000, 1000);
	background(100);
	// for (var i = 0; i < 8; i++){
	// 	print(cos(TAU*i/8), sin(TAU*i/8));
	// }
	// print(cos(PI+HALF_PI), sin(PI+HALF_PI))
	for (var i = 0; i < N_PLAYERS; i++) {
		var player = new Player();
		player.theta = random(PI+HALF_PI, TAU);
		players.push(player);
	}
}

function draw() {
	background(255,255,255);
	stroke(0,255,0);
	fill(0,255,0);
	ellipse(100+0,100+0,10,10);
	stroke(255,0,0);
	fill(255,0,0);
	ellipse(100+10000/_scale,100+10000/_scale,10,10);
	stroke(0,0,0);
	fill(0,0,0);
	//print(players[0].x, players[0].y, players[0].vx, players[0].vy, cos(players[0].theta), sin(players[0].theta))
	for (var player of players) {
		ellipse(100+player.x/_scale, 100+player.y/_scale, 8, 8);
		simulate(player);
		update(player);
  }
}
