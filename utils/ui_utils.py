import cv2
import math
import copy
import numpy as np
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .common import array_to_data
from .act_func import *


def calc_polygon(sample_np, sample_persistant, magnitude=1):
    polygon = []
    sample_np = magnitude * 50 * (sample_np - np.min(sample_persistant)) / (np.max(sample_persistant) - np.min(sample_persistant))
    for i in range(len(sample_np[0])):
        r = sample_np[0,i]
        angle = i*(360/512)
        x_poly = r * math.cos(math.radians(angle))
        y_poly = r * math.sin(math.radians(angle))
        polygon += [[x_poly, y_poly]]

    polygon = np.array(polygon, dtype = np.int32)
    polygon = polygon + 100
    return polygon

def draw_poly(polygon, window):
    polygon = copy.deepcopy(polygon)
    array_walk = np.zeros((200,200,3), dtype = np.uint8)
    array_walk = cv2.polylines(array_walk, [polygon], True, (211, 211, 211), 1)
    data_walk = array_to_data(array_walk)
    window["walk_graph"].draw_image(data=data_walk, location=(0, 200))
    return window

def find_bbox(x, y, arr):
    x_arr = np.array([x] * len(arr[:,0]))
    y_arr = np.array([y] * len(arr[:,0]))
    diff_x = np.abs(x_arr - arr[:,0])
    diff_y = np.abs(897 - y_arr - arr[:,1])
    
    id_x = np.where(diff_x == diff_x.min())[0]
    id_y = np.where(diff_y == diff_y.min())[0]
    
    ans = id_y[0]
    
    if diff_x[ans] > 50:
        ans = -1
    
    return ans

def draw_graph(window, layers_group, layers_act, node_pos, layers2disable, array_persistant):
    array = copy.deepcopy(array_persistant)
    data = array_to_data(array)
    window["viz_graph"].draw_image(data=data, location=(0, 901))


    for l in layers2disable:
        array_temp = array_persistant[node_pos[l][1] - 7: node_pos[l][1] + 7, node_pos[l][0] - 50:node_pos[l][0] + 50].copy()
        array_temp[7:9,:] = 0
        array[node_pos[l][1] - 7: node_pos[l][1] + 7, node_pos[l][0] - 50:node_pos[l][0] + 50] = array_temp

    for l in layers_group:
        array[node_pos[l][1]-7:node_pos[l][1]+7,node_pos[l][0]-50:node_pos[l][0]+50] = (255-array[node_pos[l][1]-7:node_pos[l][1]+7,node_pos[l][0]-50:node_pos[l][0]+50])


    data = array_to_data(array)
    window["viz_graph"].draw_image(data=data, location=(0, 901))

    return window

def visibility_logic(window, mode):
    # window['-CANVAS-'].update(visible=mode)
    # window['-PLOT-'].update(visible=mode)

    # window['relu'].update(visible=mode)
    # window['sin'].update(visible=mode)
    # window['cos'].update(visible=mode)
    # window['re'].update(visible=mode)
    # window['shi'].update(visible=mode)

    window['ab_graph'].update(visible=mode)
    window['graph_control'].update(visible=mode)
    window['-RESET-'].update(visible=mode)
    window['x_text'].update(visible=mode)
    window['y_text'].update(visible=mode)

    return window

def visibility_logic2(window, mode):
    window['-CANVAS-'].update(visible=mode)
    window['-PLOT-'].update(visible=mode)

    window['relu'].update(visible=mode)
    window['sin'].update(visible=mode)
    window['cos'].update(visible=mode)
    window['re'].update(visible=mode)
    window['shi'].update(visible=mode)

    window['ab_graph'].update(visible=mode)
    window['graph_control'].update(visible=mode)
    window['-RESET-'].update(visible=mode)
    window['x_text'].update(visible=mode)
    window['y_text'].update(visible=mode)

    return window


def draw_figure(canvas, figure):
    tkcanvas = FigureCanvasTkAgg(figure, canvas)
    tkcanvas.draw()
    tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    return tkcanvas



def update_viz(a, b, window, plot, fig, update_plot, block=False):
    x, y = update_plot(a, b)
    plot.set_xdata(x)
    plot.set_ydata(y)

    # ax = fig.get_axes()[0]
    # if block:
    #     ax_text = ax.text(0.1, 1.5, "DISABLED", backgroundcolor='#e5e5e5', ha="center", va="center", zorder=10, color='#1B1B1B', fontsize='xx-large', fontweight='bold')
    # elif ax_text!=None:
    #     ax_text.set_text("")

    fig.canvas.draw()
    fig.canvas.flush_events()

    # time.sleep(0.1)
    
    return window, plot, fig


def update_rad(act, window):
    if act == 0:
        window['relu'].update(False)
        window['sin'].update(True)
        window['cos'].update(False)
        window['re'].update(False)
        window['shi'].update(False)
        update_plot = update_plot_sin
    elif act == 1:
        window['relu'].update(False)
        window['sin'].update(False)
        window['cos'].update(True)
        window['re'].update(False)
        window['shi'].update(False)
        update_plot = update_plot_cos
    elif act == 2:
        window['relu'].update(False)
        window['sin'].update(False)
        window['cos'].update(False)
        window['re'].update(True)
        window['shi'].update(False)
        update_plot = update_plot_ren
    elif act == 3:
        window['relu'].update(False)
        window['sin'].update(False)
        window['cos'].update(False)
        window['re'].update(False)
        window['shi'].update(True)
        update_plot = update_plot_shi
    else:
        window['relu'].update(True)
        window['sin'].update(False)
        window['cos'].update(False)
        window['re'].update(False)
        window['shi'].update(False)
        update_plot = update_plot_relu

    return window, update_plot

