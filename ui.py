def generate(g_ema, device, mean_latent, noise, a, b, idxs, act, gain, random_seed=1):
    
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
                ab=[a,b], idx=idxs, act=act, noise = noise, gain = gain
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

def gen_rand(sz, w, h, w1, h1):
    a = torch.zeros((sz, sz))
    noise = a.new_empty((sz, sz)).normal_().numpy()
    # noise = noise[]
    noise_img = cv2.resize(np.uint8(255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))), (1024, 1024), interpolation = cv2.INTER_NEAREST)
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

    return noise_temp

def noise_window(noise, noise_torch, noise_params, gain, interaction_seq_mem):
    graph=sg.Graph(canvas_size=(1024,1024), graph_bottom_left=(0, 0), graph_top_right=(1024,1024),
    background_color='red', enable_events=True, drag_submits=True, key='graph')

    layer_colum = [[sg.Button('Randomize', key = 'R')]]

    for i in range(1, 18):
        layer_colum += [[sg.Button('Noise Injection ' + str(i), key = '-N' + str(i))]]

    layer_colum += [[sg.Button('Set Gain', key = 'set_gain')],
                    [sg.Input(size=(5, 5), key='gain')],
                    [sg.Button('Set All', key = 'set_all')],
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
                
                # noise_temp = cv2.resize(noise_temp.copy(), (1024, 1024), interpolation = cv2.INTER_NEAREST) 
                noise_new_real, noise_img = gen_rand(sz, w_ratio, h_ratio, w, h)

                noise_temp[1024-max_y:1024-min_y, min_x:max_x] = noise_img[1024-max_y:1024-min_y, min_x:max_x] 
                noise_torch[layer][0,0,min_x_ratio:max_x_ratio, min_y_ratio:max_y_ratio] = torch.from_numpy(noise_new_real[min_x_ratio:max_x_ratio, min_y_ratio:max_y_ratio]) #* gain[layer]
                
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
    return noise, noise_torch, noise_params, gain, layer, interaction_seq_mem

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

    layer_colum = [[sg.Text('Mapping Network')]]
    for i in range(8):
        layer_colum += [[sg.Button('Layer ' + str(i), key = str(i))]]
    layer_colum += [[sg.Text('Synth Network')]]
    for i in range(8, 24):
        layer_colum += [[sg.Button('Layer ' + str(i), key = str(i))]]

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
        [sg.Text('-3',key='A1'), sg.Slider(range=(-3, 3), default_value=0, resolution=0.01, enable_events=True,
         expand_x=True, orientation='horizontal', key='-A-'), sg.Text('3',key='A2')],
        [sg.Text('-3',key='B1'), sg.Slider(range=(-3, 3), default_value=0, resolution=0.01, enable_events=True,
         expand_x=True, orientation='horizontal', key='-B-'), sg.Text('3',key='B2')],
        [sg.Button('Reset', key = '-RESET-')],
        [sg.Text('a Min:'), sg.Input(size=(5, 5), key='a_min', enable_events=True, default_text='-3'), 
         sg.Text('a Max:'), sg.Input(size=(5, 5), key='a_max', enable_events=True, default_text='3')], 
        [sg.Text('b Min:'), sg.Input(size=(5, 5), key='b_min', enable_events=True, default_text='-3'), 
         sg.Text('b Max:'), sg.Input(size=(5, 5), key='b_max', enable_events=True, default_text='3')],
        [sg.Button('Set', key = 'set')]
    ]

    image_column = [
        [sg.Image('', key='-IMAGE-')],
        [sg.Text('Random Seed:'), sg.Input(size=(5, 5), enable_events=True, key='SEED', default_text='1')],
        [sg.Button('Generate', key = '-GEN-')],
        [sg.Button('Edit Noise', key = '-Noise-')],
        [sg.Button('Save', key = '-SAVE-')]
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
    gain = [1] * 18

    random_seed = 1

    timestamp = time.time()
    interaction_seq_mem = [[a_vals[:], b_vals[:], act[:], gain[:], [random_seed], [int(timestamp)]]]

    img = generate(g_ema, device, mean_latent, noise_gen, a_vals, b_vals, idxs, act, gain, random_seed)
    img_save = img_post(img)

    img = Image.fromarray(img_save)
    bio = io.BytesIO()
    img.save(bio, format= 'PNG')
    imgbytes = bio.getvalue()
    window['-IMAGE-'].Update(data=imgbytes)

    while True:
        event, values = window.read()
        if event in ['-A-', '-B-']:
            if idx not in idxs:
                idxs += [idx]
                
            a_vals[idx] = values['-A-']
            b_vals[idx] = values['-B-']
            timestamp = time.time()
            interaction_seq_mem += [[a_vals[:], b_vals[:], act[:], gain[:], [random_seed], [int(timestamp)]]]
            
            img = generate(g_ema, device, mean_latent, noise_gen, a_vals, b_vals, idxs, act, gain, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
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

            timestamp = time.time()
            interaction_seq_mem += [[a_vals[:], b_vals[:], act[:], gain[:], [random_seed], [int(timestamp)]]]
            
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
            img = generate(g_ema, device, mean_latent, noise_gen, a_vals, b_vals, idxs, act, gain, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
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
            window['relu'].update(value=True)

            a_vals[idx] = 0
            b_vals[idx] = 0

            img = generate(g_ema, device, mean_latent, noise_gen, a_vals, b_vals, idxs, act, gain, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
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

            timestamp = time.time()
            interaction_seq_mem += [[a_vals[:], b_vals[:], act[:], gain[:], [random_seed], [int(timestamp)]]]
            
        if event == 'sin':
            act[idx] = 0
            update_plot = update_plot_sin
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=False)
            window['-B-'].update(disabled=False)
            timestamp = time.time()
            interaction_seq_mem += [[a_vals[:], b_vals[:], act[:], gain[:], [random_seed], [int(timestamp)]]]
            
        if event == 'cos':
            act[idx] = 1
            update_plot = update_plot_cos
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=False)
            window['-B-'].update(disabled=False)
            timestamp = time.time()
            interaction_seq_mem += [[a_vals[:], b_vals[:], act[:], gain[:], [random_seed], [int(timestamp)]]]
            
        if event == 're':
            act[idx] = 2
            update_plot = update_plot_ren
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=False)
            window['-B-'].update(disabled=True)
            timestamp = time.time()
            interaction_seq_mem += [[a_vals[:], b_vals[:], act[:], gain[:], [random_seed], [int(timestamp)]]]
            
        if event == 'shi':
            act[idx] = 3
            update_plot = update_plot_shi
            window, act_plot, fig = update_viz(a_vals[idx], b_vals[idx], window, act_plot, fig, update_plot)
            window['-A-'].update(disabled=False)
            window['-B-'].update(disabled=True)
            timestamp = time.time()
            interaction_seq_mem += [[a_vals[:], b_vals[:], act[:], gain[:], [random_seed], [int(timestamp)]]]
            
        if event in ['set']:
            a_min = values['a_min']
            a_max = values['a_max']
            b_min = values['b_min']
            b_max = values['b_max']
            
            if '-' not in [a_min, a_max, b_min, b_max]:
                window['-A-'].update(range=(int(a_min), int(a_max)))
                window['-B-'].update(range=(int(b_min), int(b_max)))
                window['A1'].update(str(a_min))
                window['A2'].update(str(a_max))
                window['B1'].update(str(b_min))
                window['B2'].update(str(b_max))
            
        if event in ['SEED'] and values['SEED'].isnumeric():
            random_seed = int(values['SEED'])
            
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-SAVE-":
            # print('./raw/imgs/' + str(int(time.time())) + '.png')
            timestamp = time.time()
            cv2.imwrite('./raw/imgs/' + str(int(timestamp)) + '.png', img_save[..., ::-1])
            torch.save(noise_gen, './raw/noise/'+str(int(timestamp))+'.pt')
            interaction_seq = [a_vals, b_vals, act, gain, [random_seed], [int(timestamp)]]

            # with open("./raw/raw/"+str(int(timestamp))+".csv", "w") as f:
            #     wr = csv.writer(f, delimiter=" ")
            #     wr.writerows(interaction_seq)

            with open("./raw/raw/"+str(int(timestamp))+".csv", 'w') as f:
               writer = csv.writer(f)
               writer.writerows(interaction_seq)
            with open("./raw/raw/"+str(int(timestamp))+"_1.csv", 'w') as f:
               writer = csv.writer(f, delimiter='\n')
               writer.writerows(interaction_seq_mem)

        
        if event in ['-Noise-']: 
            noise_gen_np,noise_gen,noise_params,gain,layer, interaction_seq_mem = noise_window(noise_gen_np, noise_gen, noise_params, gain, interaction_seq_mem)
        
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
    import io, csv
    from PIL import Image
    import cv2

    random_seed = 1
    torch.manual_seed(random_seed)

    main()