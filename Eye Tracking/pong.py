import pygame
import sys
import random

# Set constants
WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 10
PADDLE_WIDTH, PADDLE_HEIGHT = 80, 15
BALL_SPEED = 3
PADDLE_SPEED = 3
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
FPS = 120

class Paddle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)

    def move(self, speed):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.move_ip(-speed, 0)
        if keys[pygame.K_RIGHT]:
            self.rect.move_ip(speed, 0)
        self.rect.clamp_ip(screen.get_rect())

class Ball:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x - BALL_RADIUS // 2, y - BALL_RADIUS // 2, BALL_RADIUS, BALL_RADIUS)
        self.dx = 0
        self.dy = -BALL_SPEED

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, self.rect.center, BALL_RADIUS)

    def move(self):
        self.rect.move_ip(self.dx, self.dy)

    def reset(self, x, y):
        self.rect.center = (x, y)
        self.dx = 0
        self.dy = -BALL_SPEED

    def bounce(self, paddle):
        hit_pos = (self.rect.centerx - paddle.rect.left) / PADDLE_WIDTH
        if hit_pos == 0.5:
            self.dx = random.uniform(-3, 3)*2  # Decrease the current speed by 30%
        else:
            self.dx = BALL_SPEED * (hit_pos - 0.5) * 2  # Adjust the x-speed according to hit position


class Barrier:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

player_paddle = Paddle(WIDTH // 2 - PADDLE_WIDTH // 2, HEIGHT - 2 * PADDLE_HEIGHT)
ai_paddle = Paddle(WIDTH // 2 - PADDLE_WIDTH // 2, PADDLE_HEIGHT)
ball = Ball(WIDTH // 2, HEIGHT // 2)

left_barrier = Barrier(0, 0, BALL_RADIUS, HEIGHT)
right_barrier = Barrier(WIDTH - BALL_RADIUS, 0, BALL_RADIUS, HEIGHT)

player_score = 0
ai_score = 0

font = pygame.font.Font(None, 36)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(BLACK)

    player_paddle.move(PADDLE_SPEED)
    player_paddle.draw(screen)

    if random.random() < 0.85:
        if ball.rect.centerx > ai_paddle.rect.centerx:
            ai_paddle.rect.move_ip(PADDLE_SPEED, 0)
        elif ball.rect.centerx < ai_paddle.rect.centerx:
            ai_paddle.rect.move_ip(-PADDLE_SPEED, 0)
    ai_paddle.rect.clamp_ip(screen.get_rect())
    ai_paddle.draw(screen)

    ball.move()
    ball.draw(screen)

    left_barrier.draw(screen)
    right_barrier.draw(screen)

    if ball.rect.colliderect(player_paddle.rect):
        ball.bounce(player_paddle)
        ball.dy *= -1
    elif ball.rect.colliderect(ai_paddle.rect):
        ball.bounce(ai_paddle)
        ball.dy *= -1
    elif ball.rect.colliderect(left_barrier.rect) or ball.rect.colliderect(right_barrier.rect):
        ball.dx *= -1
    elif ball.rect.top < 0:
        player_score += 1
        ball.reset(WIDTH // 2, HEIGHT // 2)
    elif ball.rect.bottom > HEIGHT:
        ai_score += 1
        ball.reset(WIDTH // 2, HEIGHT // 2)

    score_text = font.render(f"Player: {player_score} AI: {ai_score}", True, RED)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)

