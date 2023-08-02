# -*- coding: utf-8 -*-
"""
Author: Deniz PANCAR
"""
import PySimpleGUI as sg

# layout

layout = [
    [sg.Text("Eye and Gesture Tracking")],
    [sg.Button("Pong"),sg.Button("Generic Hand Game")],
    [sg.Button("Connect 4"),sg.Button("Generic Hand Game 2")]
]

window = sg.Window("Eye and Gesture Tracking", layout)

while True:
    event, values = window.read()
    print(event, values)
    if event is sg.WINDOW_CLOSED:
        break
    
    if event == "Pong":
        sg.popup_error("Not yet implemented")
    
    if event == "Generic Hand Game":
        sg.popup_error("Not yet implemented")
    
    if event == "Connect 4":
        sg.popup_error("Not yet implemented")
    
    if event == "Generic Hand Game 2":
        sg.popup_error("Not yet implemented")

window.close()