
import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import pygame 
from pygame import mixer
import random
import vidmaker

WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 10
PADDLE_WIDTH, PADDLE_HEIGHT = 80, 15
BALL_SPEED = 3
PADDLE_SPEED = 5
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
FPS = 120

# variables 
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
start_voice= False
counter_right=0
counter_left =0
counter_center =0 

video = vidmaker.Video("OutputPong.mp4", late_export=True)

# constants
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX

# initialize mixer 
mixer.init()
# loading in the voices/sounds 
voice_left = mixer.Sound('Voice/left.wav')
voice_right = mixer.Sound('Voice/Right.wav')
voice_center = mixer.Sound('Voice/center.wav')

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh

# camera object 
camera = cv.VideoCapture(0)
_, frame = camera.read()
img = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
img_hieght, img_width = img.shape[:2]
print(img_hieght, img_width)



# video Recording setup 

out = cv.VideoWriter('output1.mp4', 
                         cv.VideoWriter_fourcc(*'MP4V'),
                         10, (img_width, img_hieght))
# landmark detection function 

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    # cv.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Eyes Postion Estimator 
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    # was 130
    ret, threshed_eye = cv.threshold(median_blur, 145, 255, cv.THRESH_BINARY)

    # create fixd part for eye with 
    piece = int(w/3) 

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

# creating pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color

#pygame

class Paddle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)

    def move(self, speed, direction):
        if direction == "Left":
            self.rect.move_ip(-speed, 0)
        if direction == "Right":
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

with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        #  resizing frame
        
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            #utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

            if ratio >5.5:
                CEF_COUNTER +=1
                # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                utils.colorBackgroundText(frame,  'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

            else:
                if CEF_COUNTER>CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    CEF_COUNTER =0
            # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
            utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
            
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

            # Blink Detector Counter Completed
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            # cv.imshow('right', crop_right)
            # cv.imshow('left', crop_left)
            eye_position_right, color = positionEstimator(crop_right)
            utils.colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
            eye_position_left, color = positionEstimator(crop_left)
            utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
            
            # Starting Voice Indicator 
            if eye_position_right=="RIGHT" and pygame.mixer.get_busy()==0 and counter_right<2:
            # if eye_position_right=="RIGHT" and pygame.mixer.get_busy()==0:
    
                # starting counter 
                counter_right+=1
                # resetting counters 
                counter_center=0
                counter_left=0
                # playing voice 
                voice_right.play()
                direction = "Right"


            if eye_position_right=="CENTER" and pygame.mixer.get_busy()==0 and counter_center <2:
            # if eye_position_right=="CENTER" and pygame.mixer.get_busy()==0 :
                
                # starting Counter 
                counter_center +=1
                # resetting counters 
                counter_right=0
                counter_left=0
                # playing voice 
                voice_center.play()
                direction = "Center"
            
            if eye_position_right=="LEFT" and pygame.mixer.get_busy()==0 and counter_left<2: 
            # if eye_position_right=="LEFT" and pygame.mixer.get_busy()==0: 
                counter_left +=1
                # resetting counters 
                counter_center=0
                counter_right=0
                # playing Voice 
                voice_left.play
                direction = "Left"


        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time
        

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        # wirting the video for demo purpose 
        out.write(frame)
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        
        #pygame
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv.destroyAllWindows()
                camera.release()
                out.release()
                video.export(verbose=True)
                break

        screen.fill(BLACK)

        player_paddle.move(PADDLE_SPEED, direction)
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
        
        video.update(pygame.surfarray.pixels3d(screen).swapaxes(0, 1), inverted=False)
        
        pygame.display.flip()
        clock.tick(FPS)
