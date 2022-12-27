from math import cos, sin, radians, degrees, sqrt, pow

import pygame
from pygame.math import Vector2

import numpy as np

from sklearn.preprocessing import MinMaxScaler

pygame.init()
pygame.font.init()
pygame.display.set_caption("AI Driving")


### CONSTANTS ###

# Track
TRACK1 = ("track1", Vector2(520, 635), 3)
TRACK2 = ("track2", Vector2(570, 655), 5)
TRACK3 = ("track3", Vector2(650, 680), 3)

SELECTED_TRACK = TRACK1 # Selected Track

TRACK_NAME, INITIAL_POSITION, TRACK_LAPS = SELECTED_TRACK

# Use previous genes
LOAD_GENES = False
GENES_FILE = 'genes.txt'

# Screen
SCREEN_WIDTH = 1250
SCREEN_HEIGHT = 750
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Car
CAR_LENGTH = 32
CAR_ACCELERATION = 2.0
CAR_BREAK = -5.0
CAR_STEERING = 30
MIN_VELOCITY = 5
MAX_VELOCITY = 15

# Gene
GENE_INPUTS = 5     # 5 radars in angles -60, -30, 0, 30, 60
GENE_OUTPUTS = 9    # ( accelerate / break / nothing ) * ( steer left / steer right / nothing )
GENE_SIZE = (GENE_INPUTS, GENE_OUTPUTS)

# Algorithm
N_GENES = 100
N_BEST_GENES = 50
N_GENERATIONS = 500
RANDOM_PROBABILITY = 0.05
MUTATION_PROBABILITY = 0.20

# Images
TRACK = pygame.image.load("Assets/" + TRACK_NAME + ".png")
CAR = pygame.image.load("Assets/car.png")
BEST_CAR = pygame.image.load("Assets/best_car.png")
FONT = pygame.font.SysFont('Comic Sans MS', 20)

# Other
DRAW_COLLISION_POINTS = False
DRAW_RADARS = True
SCALER = MinMaxScaler()
GRASS_COLOR = pygame.Color(34, 177, 76, 255)
FINISH_LINE_COLOR = pygame.Color(255, 243, 0, 255)
CLOCK = pygame.time.Clock()
TICKS = 60


### CLASS DEFINITIONS ###

class Car:
    def __init__(self, image):
        self.image = image

        self.position = INITIAL_POSITION.copy()
        self.velocity = Vector2(0.0, 0.0)
        self.angle = 0.0

        self.steering = 0.0
        self.acceleration = 0.0

        self.alive = True
        self.collision_point_left = [0, 0]
        self.collision_point_right = [0, 0]
        self.update_collision_points()

    def check_collision(self):
        if SCREEN.get_at(self.collision_point_left) == GRASS_COLOR or SCREEN.get_at(self.collision_point_right) == GRASS_COLOR:
            self.alive = False

    def update_collision_points(self):
        self.collision_point_right = [
            int(self.position.x + cos(radians(self.angle + 18)) * CAR_LENGTH / 2),
            int(self.position.y - sin(radians(self.angle + 18)) * CAR_LENGTH / 2)
        ]
        
        self.collision_point_left = [
            int(self.position.x + cos(radians(self.angle - 18)) * CAR_LENGTH / 2),
            int(self.position.y - sin(radians(self.angle - 18)) * CAR_LENGTH / 2)
        ]

    def update(self):
        self.check_collision()

        # Update Velocity
        self.velocity += (self.acceleration, 0)
        self.velocity.x = max(MIN_VELOCITY, min(self.velocity.x, MAX_VELOCITY))

        # Update Steering
        if self.steering:
            turning_radius = CAR_LENGTH / sin(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        # Update Position
        self.position += self.velocity.rotate(-self.angle)
        self.angle += degrees(angular_velocity)

        self.update_collision_points()

        return self.velocity.x

    def draw(self):
        rotated = pygame.transform.rotate(self.image, self.angle)
        rect = rotated.get_rect()
        SCREEN.blit(rotated, self.position - (rect.width / 2, rect.height / 2))

        if DRAW_COLLISION_POINTS:
            pygame.draw.circle(SCREEN, (255, 0, 0, 0), self.collision_point_right, 4)
            pygame.draw.circle(SCREEN, (255, 0, 0, 0), self.collision_point_left, 4)

class AICar(Car):
    def __init__(self, gene, image):
        super().__init__(image)
        self.gene = gene
        self.score = 0.0
        self.lap = 0
        self.movements = 0
        self.lap_movements = 0
        self.distance = 0.0
        self.radars = []
        self.distances = []

    def radar(self, angle, origin):
        length = 0
        x = int(origin[0])
        y = int(origin[1])

        while SCREEN.get_at((x, y)) != GRASS_COLOR:
            length += 1
            x = int(origin[0] + cos(radians(self.angle + angle)) * length)
            y = int(origin[1] - sin(radians(self.angle + angle)) * length)

        return x, y, int(sqrt(pow(origin[0] - x, 2) + pow(origin[1] - y, 2)))

    def update_radars(self):
        self.radars = []
        self.distances = []

        # Left Radars
        for angle in [-60, -30]:
            x, y, distance = self.radar(angle, self.collision_point_left)
            self.radars.append((x, y))
            self.distances.append(distance)

        # Forward radar
        x, y, distance = self.radar(0, self.position)
        self.radars.append((x, y))
        self.distances.append(distance)

        # Right radars
        for angle in [30, 60]:
            x, y, distance = self.radar(angle, self.collision_point_right)
            self.radars.append((x, y))
            self.distances.append(distance)

    def update(self):
        distance = super().update()
        self.distance += distance
        self.movements += 1
        self.lap_movements += 1

        # if self.movements > 350: print("wtf", self.movements, self.position)

        # The first lap gives points to the further the car goes
        if self.lap < 1: self.score = self.distance

        # After the first lap the goal is to make laps faster
        if SCREEN.get_at((int(self.position[0]), int(self.position[1]))) == FINISH_LINE_COLOR and self.lap_movements > 75:
            self.lap += 1

            if self.lap >= 1: self.score = 5000 + (self.distance / self.movements) * 20 # Give more points to faster laps

            self.lap_movements = 0

            # If the car finishes the race
            if self.lap == TRACK_LAPS: 
                self.score += 5000
                self.alive = False

    def get_input(self):
        distances = np.array(self.distances).reshape(-1, 1)
        distances = SCALER.fit_transform(distances)
        return distances.reshape(1, -1)[0]

    def get_output(self):
        self.update_radars()
        return 1.0 / (1.0 + np.exp(np.dot(self.get_input(), self.gene)))

    def draw(self):
        super().draw()
        if DRAW_RADARS:
            for i, (x, y) in enumerate(self.radars):
                if i < 2: origin = self.collision_point_left
                elif i < 3: origin = self.position
                else: origin = self.collision_point_right
                pygame.draw.line(SCREEN, (255, 255, 255, 255), origin, (x, y), 1)
                pygame.draw.circle(SCREEN, (0, 255, 255, 0), (x, y), 3)


### FUNCTION DEFENITIONS ###

def best_performing_genes(scores):
    return np.argsort(scores)[-N_BEST_GENES:][::-1]

def crossover_genes(genes, best_genes):
    new_genes = np.zeros((N_GENES, GENE_INPUTS, GENE_OUTPUTS))

    for i in range(N_GENES):
        if i < N_BEST_GENES:
            new_genes[i] = np.copy(genes[best_genes[i]])
            continue
        random1, random2 = np.random.randint(N_BEST_GENES, size = 2)
        choice = np.random.randint(2, size = new_genes[random1].size).reshape(new_genes[random1].shape).astype(bool)
        new_genes[i] = np.where(choice, np.copy(new_genes[random1]), np.copy(new_genes[random2]))

    for i in range(N_BEST_GENES, N_GENES):
        random_gene = np.random.rand() < RANDOM_PROBABILITY
        if random_gene: new_genes[i] = np.random.normal(0, 1.0, size = GENE_SIZE)
    
    return new_genes

def mutate_genes(genes):
    for gene_index in range(N_BEST_GENES, N_GENES):
        for input_index in range(len(genes[gene_index])):
            for output_index in range(len(genes[gene_index][input_index])):
                random_gene = np.random.rand() < MUTATION_PROBABILITY
                if random_gene: genes[gene_index][input_index][output_index] = np.random.normal(0, 1.0, size = 1)
    
    return genes

def create_genes():
    genes = np.zeros((N_GENES, GENE_INPUTS, GENE_OUTPUTS))

    for i in range(N_GENES):
        genes[i] = np.random.normal(0, 1.0, size = GENE_SIZE)

    return genes

def cars_from_genes(genes):
    cars = [AICar(genes[0], BEST_CAR)]
    for i, gene in enumerate(genes):
        if i > 0: cars.append(AICar(gene, CAR))
    return cars

def handle_actions(car, actions):
    action = np.argmax(actions)

    if action < 3:
        car.acceleration = CAR_ACCELERATION
        if action == 0: car.steering = 0                # Accelerate
        if action == 1: car.steering = CAR_STEERING     # Accelerate and Steer Left
        if action == 2: car.steering = -CAR_STEERING    # Accelerate and Steer Right
    elif action < 6:
        car.acceleration = CAR_BREAK
        if action == 3: car.steering = 0                # Break
        if action == 4: car.steering = CAR_STEERING     # Break and Steer Left
        if action == 5: car.steering = -CAR_STEERING    # Break and Steer Right
    else:
        car.acceleration = 0
        if action == 6: car.steering = 0                # Nothing
        if action == 7: car.steering = CAR_STEERING     # Steer Left
        if action == 8: car.steering = -CAR_STEERING    # Steer Right

def run_loop(cars, generation, best_score):
    global DRAW_RADARS, DRAW_COLLISION_POINTS

    run = True
    scores = np.zeros(len(cars), dtype = int)
    alive_cars = [i for i in range(len(cars))]

    while run:
        # Event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Toggle radars
                    DRAW_RADARS = not DRAW_RADARS
                if event.key == pygame.K_c: # Toggle collision points
                    DRAW_COLLISION_POINTS = not DRAW_COLLISION_POINTS
        
        # Draw Track
        SCREEN.blit(TRACK, (0, 0))

        # Update Cars
        for index in alive_cars:
            car = cars[index]
            # Get Action
            actions = car.get_output()
            # Handle Action
            handle_actions(car, actions)
            # Update
            car.update()

        # Remove Death Cars
        for index in alive_cars[:]:
            if not cars[index].alive:
                scores[index] = cars[index].score
                alive_cars.remove(index)

        if len(alive_cars) == 0: break

        # Draw Cars
        for index in reversed(alive_cars): cars[index].draw()

        # Draw Text
        text1 = FONT.render('Generation: ' + str(generation) + '/' + str(N_GENERATIONS), False, (0, 0, 0))
        text2 = FONT.render('Cars Alive: ' + str(len(alive_cars)) + '/' + str(N_GENES), False, (0, 0, 0))
        text3 = FONT.render('Best Score: ' + str(best_score), False, (0, 0, 0))
        SCREEN.blit(text1, (SCREEN_WIDTH - text1.get_rect().width - 20, 20))
        SCREEN.blit(text2, (SCREEN_WIDTH - text2.get_rect().width - 20, 50))
        SCREEN.blit(text3, (SCREEN_WIDTH - text3.get_rect().width - 20, 80))

        pygame.display.flip()

        CLOCK.tick(TICKS)
    
    return scores.tolist()

def load_genes_from_file():
    try:
        f = open(GENES_FILE, 'r')
        contents = f.readlines()
        previous_genes = []
        for line in contents:
            score, shape, gene = line.split(';', 2)
            shape = tuple(map(int, shape.split(',')))
            gene = np.fromstring(gene.strip('[]'), sep = ' ', count = shape[0] * shape[1]).reshape(shape)
            previous_genes.append((int(score), gene))
        print('Finished parsing genes file!')
        return sorted(previous_genes, key = lambda x: x[0], reverse = True)
    except FileNotFoundError:
        print("File does not exist! Skipping...")
        return []

def save_gene(gene, score):
    f = open(GENES_FILE, 'a+')
    gene_string = str(score) + ';' + ','.join(map(str, gene.shape)) + ';' + np.array2string(gene.flatten(), separator = ' ', max_line_width = np.inf).replace('\n', '') + '\n'
    f.write(gene_string)
    f.close()

def dump_genes(generation, cars):
    f = open('dump.txt', 'a+')
    f.write("GENERATION:%d\n" % generation)
    dump_contents = [str(int(car.score)) + ';' + ','.join(map(str, car.gene.shape)) + ';' + np.array2string(car.gene.flatten(), separator = ' ', max_line_width = np.inf).replace('\n', '') + '\n' for car in cars]
    f.write(''.join(dump_contents))
    f.close()  

def gene_in_list(gene, gene_list):
    for x in gene_list:
        if np.array_equal(x[1], gene): return True
    return False


### MAIN ###

genes = create_genes()

if LOAD_GENES: 
    loaded_genes = load_genes_from_file()
    if len(loaded_genes) > 0: genes[:len(loaded_genes)] = [x[1] for x in loaded_genes]

best_gene = None
best_score = 0

for generation in range(1, N_GENERATIONS + 1):
    cars = cars_from_genes(genes)

    generation_scores = run_loop(cars, generation, best_score)

    dump_genes(generation, cars)

    if generation_scores is None: break

    best_index = generation_scores.index(max(generation_scores))
    best_score = generation_scores[best_index]
    best_gene = genes[best_index]

    print("Best score for generation", generation, ":", best_score)

    # Get top genes
    top_genes = best_performing_genes(generation_scores)

    # Perform crossover
    genes = np.copy(crossover_genes(genes, top_genes))

    # Perform mutations
    genes = mutate_genes(genes)

pygame.quit()

# Save best gene if it was not previously laoded
if best_gene is not None:
    if not LOAD_GENES or not gene_in_list(best_gene, loaded_genes): 
        save_gene(best_gene, best_score)
