import pygame
import random
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Initialize pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 600
PLAYER_SPEED = 30  # 5 times faster
BULLET_SPEED = 50
ENEMY_SPEED = 2

# Colors
SKY_BLUE = (135, 206, 235)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Plane Shooting Game")

# Detailed Plane designs
def draw_plane(surface):
    pygame.draw.polygon(surface, BLACK, [(25, 0), (0, 25), (50, 25)]) # body
    pygame.draw.polygon(surface, BLACK, [(15, 15), (25, 35), (35, 15)]) # tail wing
    pygame.draw.polygon(surface, BLACK, [(10, 10), (10, 25), (15, 25)]) # left wing
    pygame.draw.polygon(surface, BLACK, [(40, 10), (40, 25), (35, 25)]) # right wing
    pygame.draw.ellipse(surface, YELLOW, (20, 5, 10, 15)) # cockpit

def draw_enemy_plane(surface):
    pygame.draw.polygon(surface, BLACK, [(25, 50), (0, 25), (50, 25)]) # body
    pygame.draw.polygon(surface, BLACK, [(15, 35), (25, 15), (35, 35)]) # tail wing
    pygame.draw.polygon(surface, BLACK, [(10, 40), (10, 25), (15, 25)]) # left wing
    pygame.draw.polygon(surface, BLACK, [(40, 40), (40, 25), (35, 25)]) # right wing
    pygame.draw.ellipse(surface, RED, (20, 30, 10, 15)) # cockpit

def draw_boss_plane(surface):
    pygame.draw.polygon(surface, BLACK, [(50, 100), (0, 50), (100, 50)])  # body
    pygame.draw.polygon(surface, BLACK, [(30, 70), (50, 110), (70, 70)])  # tail wing
    pygame.draw.polygon(surface, BLACK, [(20, 80), (20, 50), (30, 50)])  # left wing
    pygame.draw.polygon(surface, BLACK, [(80, 80), (80, 50), (70, 50)])  # right wing
    pygame.draw.polygon(surface, BLACK, [(20, 65), (50, 40), (80, 65)])   # vertical tail moved up
    pygame.draw.ellipse(surface, RED, (40, 60, 20, 30))  # cockpit

def get_gaze_direction():
    _, frame = webcam.read()
    gaze.refresh(frame)
    
    if gaze.is_blinking():
        return "blink"
    elif gaze.is_right():
        return "RIGHT"
    elif gaze.is_left():
        return "LEFT"
    else:
        return "NEUTRAL"


def detect_blink():
    _, frame = webcam.read()
    gaze.refresh(frame)

    return gaze.is_blinking()

# Define Player, Bullet, Enemy, and Boss classes
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 50), pygame.SRCALPHA)
        draw_plane(self.image)
        self.rect = self.image.get_rect(center=(WIDTH / 2, HEIGHT - 30))
        self.last_shot_time = pygame.time.get_ticks() # Add this line

    def move(self):
        gaze_dir = get_gaze_direction()  # get the gaze direction
        if gaze_dir == "LEFT":
            self.rect.move_ip(-PLAYER_SPEED, 0)
        elif gaze_dir == "RIGHT":
            self.rect.move_ip(PLAYER_SPEED, 0)
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH

    def shoot(self, bullets, all_sprites):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > 500: # Adjust this value to control shooting speed
            bullet = Bullet(self.rect.centerx, self.rect.centery, BULLET_SPEED, YELLOW)
            bullets.add(bullet)
            all_sprites.add(bullet)
            self.last_shot_time = current_time



class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, speed, color):
        super().__init__()
        self.image = pygame.Surface((5, 15))
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = speed

    def update(self):
        self.rect.move_ip(0, -self.speed)
        if self.rect.top < 0:
            self.kill()

class Enemy(pygame.sprite.Sprite):
    def __init__(self, type):
        self.is_dead = False
        super().__init__()
        self.image = pygame.Surface((50, 50), pygame.SRCALPHA)  # Updated the surface size
        if type == "strong":
            draw_enemy_plane(self.image)  # Use the detailed plane design
            self.points = 20
        elif type == "moving":
            draw_enemy_plane(self.image)  # Use the detailed plane design
            self.points = 15
            self.move_speed = random.choice([-1, 1])
        else:
            draw_enemy_plane(self.image)  # Use the detailed plane design
            self.points = 10
        self.rect = self.image.get_rect(center=(random.randint(0, WIDTH), 0))
        self.type = type

    def update(self, enemy_bullets, all_sprites):
        global score
        self.rect.move_ip(0, ENEMY_SPEED)
        if self.type == "moving":
            self.rect.move_ip(self.move_speed, 0)
            if self.rect.left <= 0 or self.rect.right >= WIDTH:
                self.move_speed *= -1

        if self.rect.bottom > HEIGHT:
            display_message("You Lose!")
            return

        if random.random() < 0.04:  # 1% chance every frame
            self.shoot(enemy_bullets, all_sprites)

    def shoot(self, enemy_bullets, all_sprites):
        bullet = Bullet(self.rect.centerx, self.rect.centery, -BULLET_SPEED + 20, YELLOW)
        enemy_bullets.add(bullet)
        all_sprites.add(bullet)

    def hit(self):
        global score
        score += self.points
        self.kill()
        self.is_dead = True

class Boss(Enemy):
    def __init__(self):
        super().__init__("strong")
        self.image = pygame.Surface((100, 100), pygame.SRCALPHA)  # Set the size of the image surface first
        draw_boss_plane(self.image)  # Then draw the boss design
        self.rect = self.image.get_rect(center=(WIDTH / 2, 85))  # Adjusted the y-value to move boss down
        self.health = 100
        self.move_speed = random.choice([-1, 1]) * 2


    def hit(self):
        self.health -= 10
        if self.health <= 0:
            display_message("You Win!", False)  # Passing False for can_restart
            return



    def update(self, enemy_bullets, all_sprites):
    # Move horizontally
        self.rect.move_ip(self.move_speed, 0)
        if self.rect.left <= 0 or self.rect.right >= WIDTH:
            self.move_speed *= -1

        if random.random() < 0.08:
            self.shoot(enemy_bullets, all_sprites)

    def shoot(self, enemy_bullets, all_sprites):
        for _ in range(3):  # Shoot 5 bullets at once
            offset = random.randint(-50, 50)  # Offset to make bullets spread
            bullet = Bullet(self.rect.centerx + offset, self.rect.centery + self.rect.height // 2, -BULLET_SPEED + 6, RED)
            enemy_bullets.add(bullet)
            all_sprites.add(bullet)


def display_message(message, can_restart=True):  # Added can_restart parameter
    font = pygame.font.SysFont('Arial', 50)
    text = font.render(message, True, (255, 0, 0))
    screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
    pygame.display.flip()

    restart = False
    while not restart:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and can_restart:  #BURAYI AYARLA
                    restart = True
                elif event.key == pygame.K_q:
                    pygame.quit()
                    exit()


def run_game(level=1):
    pygame.init()  # Make sure to initialize pygame

    player = Player()

    all_sprites = pygame.sprite.Group()
    all_sprites.add(player)

    bullets = pygame.sprite.Group()
    enemy_bullets = pygame.sprite.Group()

    enemies = pygame.sprite.Group()

    global score
    score = 0
    level = 1

    level_duration = 60000  # 60 seconds
    level_start_time = pygame.time.get_ticks()
    enemy_spawn_time = 6000 - (level * 300)
    last_enemy_spawn_time = 0

    boss = None

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(SKY_BLUE)

        _, frame = webcam.read()
        gaze.refresh(frame)
        player.move()
        player.shoot(bullets, all_sprites)

        # Enemy spawning
        if level < 5:
            if pygame.time.get_ticks() - last_enemy_spawn_time > enemy_spawn_time:
                type = random.choice(["normal", "strong", "moving"])
                enemy = Enemy(type)
                enemies.add(enemy)
                all_sprites.add(enemy)
                last_enemy_spawn_time = pygame.time.get_ticks()
        elif level == 5 and boss is None:
            boss = Boss()
            enemies.add(boss)
            all_sprites.add(boss)
        elif level > 5:
            display_message("You Win!")
            return

        while pygame.event.poll().type != pygame.NOEVENT:
        	pass
        player.move()

        bullets.update()
        enemy_bullets.update()
        for enemy in enemies:
            enemy.update(enemy_bullets, all_sprites)

        bullet_hits = pygame.sprite.groupcollide(bullets, enemies, True, False)
        for bullet in bullet_hits:
            for enemy_hit in bullet_hits[bullet]:
                enemy_hit.hit()

            if enemy_hit.is_dead:
                score += enemy_hit.points

        player_hits = pygame.sprite.spritecollide(player, enemy_bullets, True)
        if player_hits:
            display_message("You Lose!")
            run_game(level)
            return


        all_sprites.draw(screen)

        # Display score
        font = pygame.font.SysFont('Arial', 32)
        score_text = font.render('Score: ' + str(score), True, GREEN)
        screen.blit(score_text, (WIDTH - 200, 10))


        # Display remaining time
        if level != 5:
            elapsed_time = pygame.time.get_ticks() - level_start_time
            remaining_time = (level_duration - elapsed_time) // 1000
            time_text = font.render(f"Time: {remaining_time}", True, GREEN)
            screen.blit(time_text, (10, 10))


        # Display level
        level_text = font.render(f"Level: {level}", True, WHITE)
        screen.blit(level_text, (WIDTH//2 - level_text.get_width()//2, 10))


        # Draw boss health bar
        # Draw boss health bar
        if boss:
            pygame.draw.rect(screen, BLACK, (WIDTH // 2 - 105, 40, 210, 20))
            pygame.draw.rect(screen, RED, (WIDTH // 2 - 100, 42, 2 * boss.health, 16))


        pygame.display.flip()

        pygame.time.Clock().tick(240)

        if pygame.time.get_ticks() - level_start_time > level_duration:
            level += 1
            level_start_time = pygame.time.get_ticks()
            last_enemy_spawn_time = 0
            if boss:
                boss.kill()
                boss = None
    webcam.release()
    pygame.quit()
run_game()




