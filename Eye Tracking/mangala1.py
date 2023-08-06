#!/usr/bin/python
# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import numpy as np
import sys
import math
import cv2 as cv
import mediapipe as mp
import time
import utils
from pygame import mixer
import random
import vidmaker

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

out = cv.VideoWriter('output2.mp4', 
                         cv.VideoWriter_fourcc(*'MP4V'),
                         10, (img_width, img_hieght))

video = vidmaker.Video("OutputPong.mp4", late_export=True)
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


# pygame

pygame.init()
cozunurluk=480,320
ekran_mod=HWSURFACE
baslik='Mangala'
fps=20
siyah=0,0,0
mavi=0,0,255
beyaz=255,255,255
kirmizi=255,0,0
ekran=pygame.display.set_mode(cozunurluk,ekran_mod)
pygame.display.set_caption(baslik)
font=pygame.font.Font(None,60)
font2=pygame.font.Font(None,35)
en=60
boy=60
def haznekutu(x,y):
    return pygame.Rect(x,y,en,boy)
hk1=[None]*6
hk2=[None]*6
for i in range(0,6):
    hk1[i]=haznekutu(en*(i+1),180)
    hk2[i]=haznekutu(en*(i+1),0)
def yenioyun():
    global s,oyuncu_birinci,oyunbitti
    s=([4]*6+[0])*2
    oyuncu_birinci=True
    oyunbitti=False
yenioyun()
def hazne(sayi,x,y,k):
    kboy=k*boy
    pygame.draw.rect(ekran,mavi,(x,y,en,kboy))
    pygame.draw.rect(ekran,beyaz,(x,y,en,kboy),1)
    yazi=font.render(str(sayi),1,siyah)
    yx,yy=yazi.get_size()
    ekran.blit(yazi,(x+en/2-yx/2,y+kboy/2-yy/2))
def olustur():
    hazne(s[6],0,boy,2)
    hazne(s[13],60*7,boy,2)
    for i in range(0,6):
        ii=5-i
        hazne(s[ii],en*(i+1),0,1)
        hazne(s[i+7],en*(i+1),180,1)
    if oyuncu_birinci:
        sirametin='1'
    else:
        sirametin='2'
    oyuncuyazi1=font2.render('1.',1,beyaz)
    oyuncuyazi2=font2.render('2.',1,beyaz)
    bilgi_yazi=font2.render('Hamle sirasi '+sirametin+'. oyuncuda',1,beyaz)
    bx,by=bilgi_yazi.get_size()
    bilgi_yazi2=font2.render('Y: Yeni Oyun ESC: Cikis',1,beyaz)
    bx2,by2=bilgi_yazi2.get_size()
    ekran.blit(oyuncuyazi1,((cozunurluk[0]-35)/2,145))
    ekran.blit(oyuncuyazi2,((cozunurluk[0]-35)/2,60))
    ekran.blit(bilgi_yazi,((cozunurluk[0]-bx)/2,boy*4+8))
    ekran.blit(bilgi_yazi2,((cozunurluk[0]-bx2)/2,boy*4+by+16))
def fonkoyunbitti():
    durum='Oyun bitti.'
    if s[13]>s[6]:
        durum='Oyunu 1. oyuncu kazandi.'
    elif s[13]<s[6]:
        durum='Oyunu 2. oyuncu kazandi.'
    else:
        durum='Oyun berabere.'
    bittiyazi1=font2.render(durum,1,kirmizi)
    x1,y1=bittiyazi1.get_size()
    bittiyazi2=font2.render('1. Oyuncu: '+str(s[13])+' 2. Oyuncu: '+str(s[6]),1,mavi)
    x2,y2=bittiyazi2.get_size()
    bittiyazi3=font2.render('Y: Yeni Oyun ESC: Cikis',1,beyaz)
    x3,y3=bittiyazi3.get_size()
    ekran.blit(bittiyazi1,((cozunurluk[0]-x1)/2,(cozunurluk[1]-y2)/2-y1))
    ekran.blit(bittiyazi2,((cozunurluk[0]-x2)/2,(cozunurluk[1]-y2)/2))
    ekran.blit(bittiyazi3,((cozunurluk[0]-x3)/2,(cozunurluk[1]-y2)/2+y2))
saat=pygame.time.Clock()
bitti=False

i = 0
direction = "Center"
directionDict = {
    "Left":-1,
    "Right":1,
    "Center":0
    }


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()
    # starting Video loop here.
    while not bitti:
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
        
        # pygame
        for olay in pygame.event.get():
            if olay.type==QUIT:
                bitti=True
            elif olay.type==KEYDOWN:
                if olay.key==K_ESCAPE:
                    bitti=True
            elif olay.key==K_y:
                    yenioyun()     
        
        if direction != "Center":
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            i += directionDict[direction]
            if i < 0:
                i = 0
            
            if i > 6:
                i = 6
            posx = (SQUARESIZE/2) + i * SQUARESIZE
            if turn == 0:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
            else: 
                pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
            
            time.sleep(0.5)
        
        if ratio > 5.5:
            if olay.button==1:
                if oyuncu_birinci:
                    for i in range(0,6):
                        if hk1[i].collidepoint(olay.pos):
                            gg=s[i+7]
                            if gg!=0:
                                for jj in range(1,gg+1):
                                    s[(i+7+jj)%14]+=1
                                    s[i+7]=0
                                son=(i+7+jj)%14
                                if son==13:
                                    oyuncu_birinci=True
                                else:
                                    if son in range(0,6):
                                        if s[son]%2==0:
                                            s[13]+=s[son]
                                            s[son]=0
                                    else:
                                        if s[son]==1:
                                            s[13]+=s[12-son]+1
                                            s[son]=0
                                            s[12-son]=0
                                    oyuncu_birinci=False
                                if sum(s[7:-1])==0:
                                    s[13]+=sum(s[:6])
                                    s[:6]=[0]*6
                                    oyunbitti=True
                else:
                    for i in range(0,6):
                        if hk2[i].collidepoint(olay.pos):
                            gg=s[5-i]
                            if gg!=0:
                                for jj in range(1,gg+1):
                                    s[(5-i+jj)%14]+=1
                                    s[5-i]=0
                                son=(5-i+jj)%14
                                if son==6:
                                    oyuncu_birinci=False
                                else:
                                    if son in range(7,13):
                                        if s[son]%2==0:
                                            s[6]+=s[son]
                                            s[son]=0
                                    else:
                                        if s[son]==1:
                                            s[6]+=s[12-son]+1
                                            s[son]=0
                                            s[12-son]=0
                                    oyuncu_birinci=True
                                if sum(s[:6])==0:
                                    s[6]+=sum(s[7:-1])
                                    s[7:-1]=[0]*6
                                    oyunbitti=True
        ekran.fill(siyah)
        if oyunbitti==False:
            olustur()
        else:
            fonkoyunbitti()
        pygame.display.flip()
        saat.tick(fps)
