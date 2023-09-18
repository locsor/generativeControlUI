from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import torch
import cv2
import numpy as np

import io
import pygraphviz as pgv
import ast
from PIL import Image

import time

font = ('gothic', 13)
color = '#1B1B1B'
button_color='#808080'
background_color='#d3d3d3'
scrollbar_color='#808080'

def build(plot_name):
    viz_graph=[[sg.Graph(canvas_size=(750, 901), graph_bottom_left=(0, 0), graph_top_right=(750,901),
                        background_color='red', enable_events=True, drag_submits=True, key='viz_graph')],
               [sg.Button('Disable Node', key='disable_node',font=font,button_color=button_color, tooltip='Disables selected node(s)'),
                sg.Button('Enable Node', key='restore_node',font=font,button_color=button_color, tooltip='Enables selected node(s)'),
                sg.Button('Enable All Nodes', key='restore_all_nodes',font=font,button_color=button_color, tooltip='Enables all node(s)')]]

    r0 = sg.Radio("ReLU", "gen", key='relu', default=True, enable_events=True, background_color=color, font=font, tooltip='Default activation function')
    r1 = sg.Radio("SinLU", "gen", key='sin', default=False, enable_events=True, background_color=color,font=font, tooltip='Sinu-sigmoidal Linear Unit')
    r2 = sg.Radio("CosLU", "gen", key='cos', enable_events=True, background_color=color,font=font, tooltip='Cosine Linear Unit')
    r3 = sg.Radio("ReLUN", "gen", key='re', enable_events=True, background_color=color,font=font, tooltip='Rectified Linear Unit N')
    r4 = sg.Radio("ShiLU", "gen", key='shi', enable_events=True, background_color=color,font=font, tooltip='Shifted Rectified Linear Unit')

    graph_control = [[sg.Text('a Min: -3', key='a_min_text', background_color=color, font=font), sg.Text('a Max: 3', key='a_max_text', background_color=color, font=font)],
                     [sg.Input(size=(5, 5), key='a_min', enable_events=True, default_text='-3', background_color=color,text_color='#FFFFFF'), 
                      sg.Input(size=(5, 5), key='a_max', enable_events=True, default_text='3', background_color=color,text_color='#FFFFFF')],
                     [sg.Text('b Min: -3', key='b_min_text', background_color=color, font=font), sg.Text('b Max: 3', key='b_max_text', background_color=color, font=font)], 
                     [sg.Input(size=(5, 5), key='b_min', enable_events=True, default_text='-3', background_color=color,text_color='#FFFFFF'), 
                      sg.Input(size=(5, 5), key='b_max', enable_events=True, default_text='3', background_color=color,text_color='#FFFFFF')],
                     [sg.Button('Set', key = 'set',font=font,button_color=button_color)]]

    walk_control = [
                    [sg.Text('Brush Size: 10', key='brush_size_text', background_color=color, font=font)],
                    [sg.Slider(range=(1, 32), default_value=4, expand_x=True, enable_events=True,
                               orientation='horizontal', key='brush_size',background_color=background_color,trough_color=button_color)],
                    [sg.Text('Magnitude: 1', key='magnitude_text', background_color=color, font=font)],
                    [sg.Slider(range=(0.1, 2), default_value=1, resolution=0.1, expand_x=True, enable_events=True, orientation='horizontal',
                               key='magnitude',background_color=background_color,trough_color=button_color)],
                    [sg.Button('Reset', key = 'reset-walk',font=font,button_color=button_color)]
                   ]

    plot_viewer_column = [
        [sg.Text(plot_name, key='-PLOT-', enable_events=True, background_color=color, font=font)],
        [r0, r1, r2, r3, r4],
        [sg.Text(size=(40, 1), key="-TOUT-", background_color=color, font=font)],
        [sg.Canvas(key='-CANVAS-', background_color=color)],
        [sg.Graph(canvas_size=(200, 200), graph_bottom_left=(0, 0), graph_top_right=(200,200),
                    background_color='red', enable_events=True, drag_submits=True, key='ab_graph', tooltip='Changne a and b parameters of activation functions'),
         sg.Column(graph_control, background_color=color, key='graph_control')],
        [sg.Graph(canvas_size=(200, 200), graph_bottom_left=(0, 0), graph_top_right=(200,200),
                    background_color='red', enable_events=True, drag_submits=True, key='walk_graph'), sg.Column(walk_control, background_color=color, key='graph_control')]
    ]

    image_column = [
        [sg.Image('', key='-IMAGE-', background_color=color)],
        [sg.Text('Random Seed:', background_color=color, font=font), sg.Input(size=(5, 5), enable_events=True, key='SEED', default_text='1', text_color='#FFFFFF', background_color=color)],
        [sg.Button('Change Seed', key = '-GEN-',font=font,button_color=button_color)],
        [sg.Text('Noise magnification:', background_color=color, font=font), sg.Input(size=(5, 5), enable_events=True, key='noise_set', default_text='1', text_color='#FFFFFF', background_color=color)],
        [sg.Button('Edit Noise', key = '-Noise-',font=font,button_color=button_color)],
        [sg.Button('Save', key = '-SAVE-',font=font,button_color=button_color)],
        [sg.Text('X: ', key = 'x_text', background_color=color, font=font), sg.Text('Y: ', key = 'y_text', background_color=color, font=font)],
        [sg.Button('Reset Activation Function', key = '-RESET-',font=font,button_color=button_color)],
        [sg.Button('Reset All Activation Functions', key = '-RESETALL-',font=font,button_color=button_color)]
    ]

    layout = [
        [   
            sg.Column(viz_graph, background_color=color),
            sg.VSeperator(),
            sg.Column(plot_viewer_column, background_color=color),
            sg.VSeperator(),
            sg.Column(image_column, background_color=color),
        ]
    ]

    return layout

def viz_prep(a_vals, b_vals, act, g_ema):
    truncation = 1

    def print_to_string(*args, **kwargs):
        output = io.StringIO()
        print(*args, file=output, **kwargs)
        contents = output.getvalue()
        output.close()
        return contents

    def generate_dry(g_ema, mean_latent, noise, a, b, idxs, act, gain, random_seed=1, dry=True):
        with torch.no_grad():
            g_ema.eval()

            torch.manual_seed(random_seed)
            sample_z = torch.randn(1, 512).cuda()
            
            data = g_ema([sample_z], truncation=1, truncation_latent=mean_latent,
                              ab=[a,b], idx=idxs, act=act, noise = noise, gain = gain, dry=dry)
            
        return data


    if truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(truncation_mean)
    else:
        mean_latent = None

    gain = [1] * 18

    random_seed = 1

    noise_gen = []
    noise_gen_np = []
    noise_params = []
    for i in [4 , 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]:
        noise = torch.zeros((1, 1, i, i)).new_empty((1, 1, i, i)).normal_()
        noise_np = noise.clone().numpy()
        noise_params += [[np.min(noise_np), np.max(noise_np), i]]
        noise_np = np.uint8(255 * (noise_np - np.min(noise_np)) / (np.max(noise_np) - np.min(noise_np)))
        noise_gen += [noise] 
        noise_gen_np += [noise_np]

    out = generate_dry(g_ema, mean_latent, noise_gen, a_vals, b_vals, [], act, gain, random_seed=1)

    str1 = [print_to_string(aa[0]).split(' ') for aa in out]
    str2 = [item for sublist in str1 for item in sublist]

    layers = []

    lin_ct = 0
    conv_ct = 0
    rgb_ct = 0

    for s in str2:
        if 'EqualLinear' in s:
            layers += [s.split('(')[0] + str(lin_ct)]
            lin_ct += 1
        elif 'StyledConv' in s:
            layers += [s.split('(')[0] + str(conv_ct)]
            conv_ct += 1
        elif 'ToRGB' in s:
            layers += [s.split('(')[0] +str(rgb_ct)]
            rgb_ct += 1
        elif 'PixelNorm' in s:
            layers += [s.split('(')[0]]

    layers[0:0] = ['Z']
    layers[10:10] = ['W']
    del layers[1]

    G = pgv.AGraph(strict=False, directed=True, landscape="false", ranksep="0.18")
    G.graph_attr['dpi'] = '50'
    edges = []

    ct = 0
    for layer in layers:
        if 'StyledConv' in layer or 'ToRGB' in layer:
            if ct-1 == 9:
                ct += 1
                continue
            edges += [[ct-1, ct, 9]]
        else:
            edges += [[ct-1, ct]]
            
        ct += 1

    edges = edges[1:]
    sz = 0
    ct = 0

    for layer in layers:
        G.add_node(layer)
        n=G.get_node(layer)
        n.attr['shape']="box"
        n.attr['height']=0.3
        n.attr['width']=2.0
        
    for edge in edges:
        
        edge0 = edge[0]
        edge1 = edge[1]
        if len(edge) == 2:
            G.add_edge(layers[edge0], layers[edge1])
        else:
            edge2 = edge[2]
            G.add_edge(layers[edge0], layers[edge1])
            G.add_edge(layers[edge2], layers[edge0])
            
    G.layout(prog="dot")

    node_pos = []
    for layer in layers:
        node = G.get_node(layer)
        node_pos += [ast.literal_eval(node.attr['pos'])]

    node_pos = np.array(node_pos)
    node_pos[:,0] /= 1.4297188755
    node_pos[:,1] /= 1.43032329989
    node_pos[:,1] = 897 - node_pos[:,1]
    node_pos = np.uint16(np.rint(node_pos))

    return node_pos, layers

def noise_window(noise, noise_torch, noise_params, gain, interaction_seq_mem):
    def set_noise(noise, layer, window, img_draw, area):
        noise_temp = cv2.resize(noise[layer][0,0], (1024, 1024), interpolation = cv2.INTER_NEAREST)
        img = Image.fromarray(noise_temp)
        bio = io.BytesIO()
        img.save(bio, format= 'PNG')
        imgbytes = bio.getvalue()

        window['graph'].delete_figure(img_draw)
        img_draw = window['graph'].draw_image(data=imgbytes, location=(0, 1024))

        if area:
            window['graph'].delete_figure(area)

        return noise_temp

    
    def gen_rand(sz, w, h, w1, h1):
        a = torch.zeros((sz, sz))
        noise = a.new_empty((sz, sz)).normal_().numpy()
        # noise = noise[]
        noise_img = cv2.resize(np.uint8(255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))), (1024, 1024), interpolation = cv2.INTER_NEAREST)
        return noise, noise_img

    graph=sg.Graph(canvas_size=(1024,1024), graph_bottom_left=(0, 0), graph_top_right=(1024,1024),
    background_color='red', enable_events=True, drag_submits=True, key='graph')

    layer_colum = [[sg.Button('Randomize', key = 'R',font=font,button_color=button_color)]]

    for i in range(1, 18):
        layer_colum += [[sg.Button('Noise Injection ' + str(i), font=font, button_color=button_color, key = '-N' + str(i))]]

    layer_colum += [[sg.Button('Set Gain', key = 'set_gain',font=font,button_color=button_color)],
                    [sg.Input(size=(5, 5), key='gain')],
                    [sg.Button('Set All', key = 'set_all',font=font,button_color=button_color)],
                    [sg.Input(size=(5, 5), key='gains')]]

    layout = [[sg.Column(layer_colum), graph]]
    window = sg.Window('Graph test', layout, finalize=True)#.bind("<ButtonRelease-1>", ' Release')
    
    img = Image.fromarray(noise[-1][0,0])
    bio = io.BytesIO()
    img.save(bio, format= 'PNG')
    imgbytes = bio.getvalue()

    img_draw = window['graph'].draw_image(data=imgbytes, location=(0, 1024))

    first_cord = []
    last_cord = []
    area = None
    
    layer = -1
    
    noise_temp = noise[-1][0,0]
    last_cord_mem = None
    first_cord_mem = None
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == 'graph':
            x, y = values['graph']
            if first_cord == []:
                first_cord = (x, y)
            last_cord = (x, y)

            if area:
                window['graph'].delete_figure(area)
            area = graph.DrawRectangle(first_cord, last_cord, line_color="white")

        if event == 'graph+UP':
            first_cord_mem = first_cord
            last_cord_mem = last_cord
            first_cord = []
            last_cord = []

        if event == 'R':
            if last_cord_mem:
                sz = noise_params[layer][2]
                
                min_x = min(first_cord_mem[0], last_cord_mem[0])
                min_y = min(first_cord_mem[1], last_cord_mem[1])

                max_x = max(first_cord_mem[0], last_cord_mem[0])
                max_y = max(first_cord_mem[1], last_cord_mem[1])

                if any(coord in [0, None] for coord in [min_x,min_y,max_x,max_y]):
                    continue
                
                ratio = 1024//sz

                w, h = max_x - min_x, max_y - min_y

                min_x_ratio = min_x//ratio
                min_y_ratio = min_y//ratio
                max_x_ratio = max_x//ratio
                max_y_ratio = max_y//ratio

                w_ratio = max_x_ratio - min_x_ratio
                h_ratio = max_y_ratio - min_y_ratio
                
                noise_new_real, noise_img = gen_rand(sz, w_ratio, h_ratio, w, h)

                noise_temp[1024-max_y:1024-min_y, min_x:max_x] = noise_img[1024-max_y:1024-min_y, min_x:max_x] 
                noise_torch[layer][0,0,min_x_ratio:max_x_ratio, min_y_ratio:max_y_ratio] = torch.from_numpy(noise_new_real[min_x_ratio:max_x_ratio, min_y_ratio:max_y_ratio]) #* gain[layer]
                
                img = Image.fromarray(noise_temp)
                bio = io.BytesIO()
                img.save(bio, format= 'PNG')
                imgbytes = bio.getvalue()

                window['graph'].delete_figure(img_draw)
                img_draw = window['graph'].draw_image(data=imgbytes, location=(0, 1024))
                
                noise_params[layer][0] = np.min(noise_torch[layer].cpu().numpy())
                noise_params[layer][1] = np.max(noise_torch[layer].cpu().numpy())
                noise_params[layer][2] = noise_torch[layer].shape[-1]
                
                noise[layer] = noise_temp[np.newaxis, np.newaxis]
            
            
        if event[:2] == '-N':
            layer = int(event[2:])-1
            # noise_temp = noise[layer][0,0]
            noise_temp = set_noise(noise, layer, window, img_draw, area)
            window['gain'].update(gain[layer])


        if event == "set_gain":
            gain[layer] = float(values["gain"]) 
            timestamp = time.time()
            interaction_seq_mem += [interaction_seq_mem[-1][:]]
            interaction_seq_mem[-1][3] = gain
            interaction_seq_mem[-1][-1] = int(timestamp)

        if event == "set_all":
            gain = [float(values["gains"]) for g in gain]
            timestamp = time.time()
            interaction_seq_mem += [interaction_seq_mem[-1][:]]
            interaction_seq_mem[-1][3] = gain
            interaction_seq_mem[-1][-1] = int(timestamp)

            
    window.close()
    return noise, noise_torch, noise_params, gain, interaction_seq_mem