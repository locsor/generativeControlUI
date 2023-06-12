def generate(g_ema, device, mean_latent, noise, a, b, idxs, act, random_seed=1):
    
    sample = 1
    pics = 1
    latent = 512
    truncation = 1

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(pics)):
            torch.manual_seed(random_seed)
            sample_z = torch.randn(sample, latent, device=device)
            
            sample, _ = g_ema(
                [sample_z], truncation=truncation, truncation_latent=mean_latent,
                ab=[a,b], idx=idxs, act=act, noise = noise
            )
        
        if pics == 1:
            return sample


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def draw_figure(canvas, figure):
    tkcanvas = FigureCanvasTkAgg(figure, canvas)
    tkcanvas.draw()
    tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    return tkcanvas

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def update_plot_relu(a, b):
    a = 0.1
    xs = np.linspace(-5, 5, num=1000)
    ys = np.zeros(1000)
    for i, x in enumerate(xs):
        ys[i] = np.maximum(a*x, x)
        
    return xs, ys

def update_plot_sin(a, b):
    xs = np.linspace(-5, 5, num=1000)
    ys = np.zeros(1000)
    for i, x in enumerate(xs):
        ys[i] = sigmoid(x) * (x + a*np.sin(b*x))
        
    return xs, ys

def update_plot_cos(a, b):
    xs = np.linspace(-5, 5, num=1000)
    ys = np.zeros(1000)
    for i, x in enumerate(xs):
        ys[i] = sigmoid(x) * (x + a*np.cos(b*x))
        
    return xs, ys

def update_plot_ren(a, b):
    xs = np.linspace(-5, 5, num=1000)
    ys = np.zeros(1000)
    for i, x in enumerate(xs):
        t = max(0.0, x)
        ys[i] = min(t, 1 * a)
        
    return xs, ys

def update_plot_shi(a, b):
    xs = np.linspace(-5, 5, num=1000)
    ys = np.zeros(1000)
    for i, x in enumerate(xs):
        ys[i] = a * np.maximum(x, 0) + b
        
    return xs, ys

def img_post(img):
    img = torch.clamp(img, min=-1, max=1)
    img = img * 0.5 + 0.5
    img = np.uint8(img[0].cpu().numpy().transpose(1, 2, 0) * 255)
    img = cv2.resize(img, (512,512))
    return img

def update_viz(a, b, window, plot, fig, update_plot):
    x, y = update_plot(a, b)
    plot.set_xdata(x)
    plot.set_ydata(y)

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)
    
    return window, plot, fig

def gen_rand(w, h, w1, h1):
    a = torch.zeros((w, h))
    noise = a.new_empty((w, h)).normal_().numpy()
    noise_img = cv2.resize(np.uint8(255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))), (w1, h1), interpolation = cv2.INTER_NEAREST)
    return noise, noise_img

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

def noise_window(noise, noise_torch, noise_params):
    graph=sg.Graph(canvas_size=(1024,1024), graph_bottom_left=(0, 0), graph_top_right=(1024,1024),
    background_color='red', enable_events=True, drag_submits=True, key='graph')

    layer_colum = [[sg.Button('Randomize', key = 'R')],
                    [sg.Button('Noise Injection 1', key = 'N1')],
                    [sg.Button('Noise Injection 2', key = 'N2')],
                    [sg.Button('Noise Injection 3', key = 'N3')],
                    [sg.Button('Noise Injection 4', key = 'N4')],
                    [sg.Button('Noise Injection 5', key = 'N5')],
                    [sg.Button('Noise Injection 6', key = 'N6')],
                    [sg.Button('Noise Injection 7', key = 'N7')],
                    [sg.Button('Noise Injection 8', key = 'N8')],
                    [sg.Button('Noise Injection 9', key = 'N9')],
                    [sg.Button('Noise Injection 10', key = 'N10')],
                    [sg.Button('Noise Injection 11', key = 'N11')],
                    [sg.Button('Noise Injection 12', key = 'N12')],
                    [sg.Button('Noise Injection 13', key = 'N13')],
                    [sg.Button('Noise Injection 14', key = 'N14')],
                    [sg.Button('Noise Injection 15', key = 'N15')],
                    [sg.Button('Noise Injection 16', key = 'N16')],
                    [sg.Button('Noise Injection 17', key = 'N17')]]
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
            sz = noise_params[layer][2]
            
            min_x = min(first_cord_mem[0], last_cord_mem[0])
            min_y = min(first_cord_mem[1], last_cord_mem[1])

            max_x = max(first_cord_mem[0], last_cord_mem[0])
            max_y = max(first_cord_mem[1], last_cord_mem[1])
            
            w, h = max_x - min_x, max_y - min_y

            ratio = 1024//sz
            min_x_ratio = min_x//ratio
            min_y_ratio = min_y//ratio
            max_x_ratio = max_x//ratio
            max_y_ratio = max_y//ratio
            
            noise_old = noise_temp.copy() 
            noise_new_real, noise_img = gen_rand(max_y_ratio-min_y_ratio, max_x_ratio-min_x_ratio, w, h)
            noise_temp[1024-max_y:1024-min_y, min_x:max_x] = noise_img
            print(noise_torch[layer].mean())
            noise_torch[layer][0,0,sz-(max_y_ratio):sz-(min_y_ratio), min_x_ratio:max_x_ratio] = torch.from_numpy(noise_new_real)
            print(noise_torch[layer].mean())
            print(sz-(max_y//ratio), sz-(min_y/ratio), min_x//ratio, max_x//ratio, ratio)
            print(noise_new_real, noise_new_real.shape)
            print(noise_torch[layer])
            
            
            img = Image.fromarray(noise_temp)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()

            window['graph'].delete_figure(img_draw)
            img_draw = window['graph'].draw_image(data=imgbytes, location=(0, 1024))
            
#             mx = noise_params[layer][1]
#             mn = noise_params[layer][0]
#             noise_temp_norm = cv2.resize(noise_temp.copy(), (sz, sz)) * (mx - mn) + mn
#             noise_torch[layer] = torch.from_numpy(noise_temp_norm).unsqueeze(0).unsqueeze(0)
            noise_params[layer][0] = np.min(noise_torch[layer].cpu().numpy())
            noise_params[layer][1] = np.max(noise_torch[layer].cpu().numpy())
            noise_params[layer][2] = noise_torch[layer].shape[-1]
            
            noise[layer] = noise_temp[np.newaxis, np.newaxis]
            print(noise[layer].shape)
            
            
        if event == 'N1':
            layer = int(event[1:])-1
            set_noise(noise, layer, window, img_draw, area)
            
    window.close()
    return noise, noise_torch, noise_params, layer

def main():
    device = "cuda"
    size = 1024
    truncation = 1
    truncation_mean = 4096
    ckpt = "./weights/stylegan2-ffhq-config-f.pt"
    channel_multiplier = 2
    latent = 512
    n_mlp = 8
        
    g_ema = Generator(
        size, latent, n_mlp, channel_multiplier=channel_multiplier).to(device)
    checkpoint = torch.load(ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"])

    if truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(truncation_mean)
    else:
        mean_latent = None
        
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

    matplotlib.use('TkAgg')

    layer_colum = [
        [sg.Text('Mapping Network')], 
        [sg.Button('Layer 1', key = '0')], 
        [sg.Button('Layer 2', key = '1')],
        [sg.Button('Layer 3', key = '2')],
        [sg.Button('Layer 4', key = '3')],
        [sg.Button('Layer 5', key = '4')],
        [sg.Button('Layer 6', key = '5')], 
        [sg.Button('Layer 7', key = '6')],
        [sg.Button('Layer 8', key = '7')],
        [sg.Text('Synth Network')], 
        [sg.Button('Block 1_1', key = '8')], 
        [sg.Button('Block 1_2', key = '9')], 
        [sg.Button('Block 2_1', key = '10')], 
        [sg.Button('Block 2_2', key = '11')], 
        [sg.Button('Block 3_1', key = '12')], 
        [sg.Button('Block 3_2', key = '13')], 
        [sg.Button('Block 4_1', key = '14')], 
        [sg.Button('Block 4_2', key = '15')], 
        [sg.Button('Block 5_1', key = '16')], 
        [sg.Button('Block 5_2', key = '17')], 
        [sg.Button('Block 6_1', key = '18')], 
        [sg.Button('Block 6_2', key = '19')], 
        [sg.Button('Block 7_1', key = '20')], 
        [sg.Button('Block 7_2', key = '21')], 
        [sg.Button('Block 8_1', key = '22')], 
        [sg.Button('Block 8_2', key = '23')], 
    ]

    plot_name = "Activation Function Plot, Layer: "

    r0 = sg.Radio("ReLU", "gen", key='relu', default=True, enable_events=True)
    r1 = sg.Radio("SinLU", "gen", key='sin', enable_events=True)
    r2 = sg.Radio("CosLU", "gen", key='cos', enable_events=True)
    r3 = sg.Radio("ReLUN", "gen", key='re', enable_events=True)
    r4 = sg.Radio("ShiLU", "gen", key='shi', enable_events=True)

    plot_viewer_column = [
        [sg.Text(plot_name, key='-PLOT-', enable_events=True)],
        [r0, r1, r2, r3, r4],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Canvas(key='-CANVAS-')],
        [sg.Text('-3',key='A1'), sg.Slider(range=(-3, 3), default_value=0, resolution=0.1, enable_events=True,
         expand_x=True, orientation='horizontal', key='-A-'), sg.Text('3',key='A2')],
        [sg.Text('-3',key='B1'), sg.Slider(range=(-3, 3), default_value=0, resolution=0.1, enable_events=True,
         expand_x=True, orientation='horizontal', key='-B-'), sg.Text('3',key='B2')],
        [sg.Button('Reset', key = '-RESET-')],
        [sg.Text('a Min:'), sg.Input(size=(5, 5), key='a_min', enable_events=True, default_text='-3'), 
         sg.Text('a Max:'), sg.Input(size=(5, 5), key='a_max', enable_events=True, default_text='3')], 
        [sg.Text('b Min:'), sg.Input(size=(5, 5), key='b_min', enable_events=True, default_text='-3'), 
         sg.Text('b Max:'), sg.Input(size=(5, 5), key='b_max', enable_events=True, default_text='3')],
        [sg.Text('Step A: '), sg.Input(size=(5, 5), key='stepA', enable_events=True, default_text='0.1')],
        [sg.Text('Step B: '), sg.Input(size=(5, 5), key='stepB', enable_events=True, default_text='0.1')],
        [sg.Button('Set', key = 'set')]
    ]

    image_column = [
        [sg.Image('', key='-IMAGE-')],
        [sg.Text('Random Seed:'), sg.Input(size=(5, 5), enable_events=True, key='SEED', default_text='1')],
        [sg.Button('Generate', key = '-GEN-')],
        [sg.Button('Edit Noise', key = '-Noise-')]
    ]

    layout = [
        [
            sg.Column(layer_colum),
            sg.VSeperator(),
            sg.Column(plot_viewer_column),
            sg.VSeperator(),
            sg.Column(image_column),
        ]
    ]

    window = sg.Window("Image Viewer", layout, size=(1500, 900), finalize=True)

    window['-A-'].update(disabled=True)
    window['-B-'].update(disabled=True)

    a_vals, b_vals = [0] * 24, [0] * 24
    a, b = 0, 0

    # Run the Event Loop
    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    x, y = update_plot_relu(a, b)
    ax = fig.add_subplot(111)
    ax.set_ylim([-2, 5])
    act_plot, = ax.plot(x, y)
    tkcanvas = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    idx = 0
    button_keys = [str(i) for i in range(24)]

    idxs = []

    update_plot = update_plot_sin
    act = [-1] * 8 + [-1]*16

    random_seed = 1

    while True:
        event, values = window.read()
        if event in ['-A-', '-B-']:
            if idx not in idxs:
                idxs += [idx]
                
            a_vals[idx] = values['-A-']
            b_vals[idx] = values['-B-']
            
            img = generate(g_ema, device, mean_latent, noise_gen, a_vals, b_vals, idxs, act, random_seed)
            img = img_post(img)

            img = Image.fromarray(img)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)
            time.sleep(0.1)
            
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            
        if event in button_keys:
            window['-PLOT-'].update(plot_name + event)
            idx = int(event)
            
            window['-A-'].update(value = a_vals[idx])
            window['-B-'].update(value = b_vals[idx])
            
            if act[idx] == 0:
                window['relu'].update(False)
                window['sin'].update(True)
                window['cos'].update(False)
                window['re'].update(False)
                window['shi'].update(False)
                update_plot = update_plot_sin
                window['-A-'].update(disabled=False)
                window['-B-'].update(disabled=False)
            elif act[idx] == 1:
                window['relu'].update(False)
                window['sin'].update(False)
                window['cos'].update(True)
                window['re'].update(False)
                window['shi'].update(False)
                update_plot = update_plot_cos
                window['-A-'].update(disabled=False)
                window['-B-'].update(disabled=False)
            elif act[idx] == 2:
                window['relu'].update(False)
                window['sin'].update(False)
                window['cos'].update(False)
                window['re'].update(True)
                window['shi'].update(False)
                update_plot = update_plot_ren
                window['-A-'].update(disabled=False)
                window['-B-'].update(disabled=True)
            elif act[idx] == 3:
                window['relu'].update(False)
                window['sin'].update(False)
                window['cos'].update(False)
                window['re'].update(False)
                window['shi'].update(True)
                update_plot = update_plot_shi
                window['-A-'].update(disabled=False)
                window['-B-'].update(disabled=True)
            else:
                window['relu'].update(True)
                window['sin'].update(False)
                window['cos'].update(False)
                window['re'].update(False)
                window['shi'].update(False)
                update_plot = update_plot_relu
                window['-A-'].update(disabled=True)
                window['-B-'].update(disabled=True)
                
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            
        if event == '-GEN-':
            img = generate(g_ema, device, mean_latent, noise_gen, a_vals, b_vals, idxs, act, random_seed)
            img = img_post(img)

            img = Image.fromarray(img)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)
            
        if event == '-RESET-':
            act[idx] = -1
            update_plot = update_plot_relu
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=True, value = 0.0)
            window['-B-'].update(disabled=True, value = 0.0)

            img = generate(g_ema, device, mean_latent, noise_gen, a_vals, b_vals, idxs, act, random_seed)
            img = img_post(img)

            img = Image.fromarray(img)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)

        if event == 'relu':
            act[idx] = -1
            update_plot = update_plot_relu
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=True)
            window['-B-'].update(disabled=True)
            
        if event == 'sin':
            act[idx] = 0
            update_plot = update_plot_sin
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=False)
            window['-B-'].update(disabled=False)
            
        if event == 'cos':
            act[idx] = 1
            update_plot = update_plot_cos
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=False)
            window['-B-'].update(disabled=False)
            
        if event == 're':
            act[idx] = 2
            update_plot = update_plot_ren
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=False)
            window['-B-'].update(disabled=True)
            
        if event == 'shi':
            act[idx] = 3
            update_plot = update_plot_shi
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=False)
            window['-B-'].update(disabled=True)
            
        if event in ['set']:
            a_min = values['a_min']
            a_max = values['a_max']
            b_min = values['b_min']
            b_max = values['b_max']
            stepA = values['stepA']
            stepB = values['stepB']
            
            if '-' not in [a_min, a_max, b_min, b_max, stepA, stepB]:
                window['-A-'].update(range=(int(a_min), int(a_max)), resolution=stepA)
                window['-B-'].update(range=(int(b_min), int(b_max)), resolution=stepB)
                window['A1'].update(str(a_min))
                window['A2'].update(str(a_max))
                window['B1'].update(str(b_min))
                window['B2'].update(str(b_max))
            
        if event in ['SEED'] and values['SEED'].isnumeric():
            random_seed = int(values['SEED'])
            
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if event in ['-Noise-']: 
            noise_gen_np,noise_gen,noise_params,layer = noise_window(noise_gen_np, noise_gen, noise_params)
            print(len(noise_gen_np)) 
        
    window.close()


if __name__ == '__main__':
    import argparse

    import torch
    from torchvision import utils
    from model import Generator
    from tqdm import tqdm

    import ipywidgets as widgets

    import numpy as np
    import time, math
    import matplotlib.pyplot as plt

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import PySimpleGUI as sg
    import matplotlib
    import os.path
    import io
    from PIL import Image
    import cv2

    random_seed = 1
    torch.manual_seed(random_seed)

    main()