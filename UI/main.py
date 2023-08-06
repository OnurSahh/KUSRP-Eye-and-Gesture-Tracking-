# -*- coding: utf-8 -*-
"""
Author: Deniz PANCAR
"""
import PySimpleGUI as sg
import subprocess
from pathlib import Path
import sys
from random import choice
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root)) # adds main directory to paths
# layout

path = Path(__file__).parents[0]

sg.set_options(scaling=1.334646962233169*5)


def get_scaling():
    # called before window created
    root = sg.tk.Tk()
    scaling = root.winfo_fpixels('1i')/72
    root.destroy()
    return scaling

pad1 = 85
pad2 = 25

layout = [
    [sg.Text("Eye and Gesture Tracking", pad=(pad1*.5, pad2))],
    [sg.Button("Random Game", pad=(pad1*1, pad2)), sg.Button("Guidelines", pad=(pad1*3, pad2), size=(10, 2))],
    [sg.Button("Pong", pad=(pad1*1, pad2), size=(10, 2)),sg.Button("Rock Paper Scissors", pad=(pad1*3, pad2), size=(13, 2))],
    [sg.Button("Connect 4", pad=(pad1*1, pad2), size=(10, 3)),sg.Button("Memory Game", pad=(pad1*3, pad2), size=(12, 3))]
]

window = sg.Window("Eye and Gesture Tracking", layout, resizable=True, margins=(0, 0)).finalize()
my_scaling = get_scaling()      # call get_scaling()
my_width, my_height = window.get_screen_size()  # call sg.Window.get_screen_size()
window.maximize()

gameLaunches = {
    "Pong":"Pong1.py",    
    "Connect 4":"Connect 4.py",
    "Rock Paper Scissors":"rock_paper_scissors.py",
    "Memory Game":"Memory_Game_Integrated.py"
}



def openGuidelines():
    column = [[sg.Image(filename='guidelines.png', key='Image')]]
    layout = [[sg.Column(column, size=(1500, 860), scrollable=True, key='Column')]]
    window = sg.Window('test', layout, finalize=True)
    window.maximize()

    while True:
        event, _ = window.read()
        if event == sg.WINDOW_CLOSED:
            break

    window.close()

while True:
    event, values = window.read()
    if event is sg.WINDOW_CLOSED:
        break
    
    if event in gameLaunches:
        gamePath = gameLaunches[event]
        if gamePath:
            with open(path / gamePath) as f:
                exec(f.read())
            break
        else:
            sg.popup_error("Not yet implemented")
    
    if event == "Random Game":
        gamePath = choice(list(gameLaunches.values()))
        if gamePath:
            with open(path / gamePath) as f:
                exec(f.read())
            break
        else:
            sg.popup_error("Not yet implemented")
    
    if event == "Guidelines":
        openGuidelines()

window.close()