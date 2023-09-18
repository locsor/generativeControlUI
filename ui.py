global name_it
name_it = 0
def generate(g_ema, sample_z, mean_latent, noise, a, b, idxs, act, gain, dis, random_seed=1):
    with torch.no_grad():
        g_ema.eval()

        torch.manual_seed(random_seed)
        
        sample, _ = g_ema(
            [sample_z], truncation=cfg.MODEL.truncation, truncation_latent=mean_latent,
            ab=[a,b], idx=idxs, act=act, noise = noise, gain = gain, disable = dis
        )
    
        global name_it
        name_it += 1
        img_post_save(sample, name_it)
        return sample


def check_rgb(ans, node_names):
    res = True
    for a in ans:
        print(node_names[a])
        if 'RGB' not in node_names[a]:
            res = False

    return res

def main():
    g_ema = Generator(cfg.MODEL.size, cfg.MODEL.latent, cfg.MODEL.n_mlp, channel_multiplier=cfg.MODEL.channel_multiplier).to(cfg.MODEL.device)
    checkpoint = torch.load(cfg.MODEL.ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"])

    if cfg.MODEL.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(cfg.MODEL.truncation_mean)
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

    a_vals = b_vals = act = np.array(cfg.MODEL.layer_markings)
    a_vals[a_vals>-1] = 0
    b_vals[b_vals>-1] = 0
    a_vals[a_vals==-1] = -1000
    b_vals[b_vals==-1] = -1000

    a_vals = np.array(a_vals, dtype=np.float16)
    b_vals = np.array(b_vals, dtype=np.float16)

    act[act==-1] = -2
    act[act>-1] = -1

    gain = [1] * 17

    brush_size = 4
    magnitude = 1

    layers_act = list(range(1,9)) + [10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 35]
    layers2disable = []

    ans = 1

    start = True

    act_names = ['relu', 'sin', 'cos', 're', 'shi']

    a_min, a_max = b_min, b_max = -3, 3

    node_pos, node_names = viz_prep(a_vals, b_vals, act, g_ema)
    # print(node_pos)
    # node_names = ['Z', 'EqualLinear0', 'EqualLinear1', 'EqualLinear2', 'EqualLinear3', 'EqualLinear4', 'EqualLinear5', 'EqualLinear6', 'EqualLinear7', 'W',
    #               'StyledConv0', 'ToRGB0', 'StyledConv1', 'StyledConv2', 'ToRGB1', 'StyledConv3', 'StyledConv4', 'ToRGB2', 'StyledConv5', 'StyledConv6', 'ToRGB3',
    #               'StyledConv7', 'StyledConv8', 'ToRGB4', 'StyledConv9', 'StyledConv10', 'ToRGB5', 'StyledConv11', 'StyledConv12', 'ToRGB6', 'StyledConv13',
    #               'StyledConv14', 'ToRGB7', 'StyledConv15', 'StyledConv16', 'ToRGB8']
    # im = Image.open("model_image.png")
    # im = #Image.open("model_io.drawio.png")
    graph_image = cv2.imread("model_image.png", 0) #np.array(im, dtype=np.uint8)
    graph_image = 255-graph_image
    graph_image[graph_image==0] = 27
    graph_image = cv2.resize(graph_image, (750,901))
    graph_image_persistant = copy.deepcopy(graph_image)
    data = array_to_data(graph_image)   

    plot_name = "Activation Function Plot, Layer: "
    layout = build(plot_name)

    font = ('gothic', 13)
    color = '#1B1B1B'
    window = sg.Window("Image Viewer", layout, size=(1850, 1200), finalize=True, return_keyboard_events=True, background_color=color)

    window["viz_graph"].draw_image(data=data, location=(0, 901))

    array_sq = cv2.line(np.zeros((200,200), dtype=np.uint8), (100,0), (100,200), (211,211,211), 2)
    array_sq = cv2.line(array_sq, (0,100), (200,100), (211,211,211), 2)
    array_sq_persistant = array_sq.copy()
    data_sq = array_to_data(array_sq)
    window["ab_graph"].draw_image(data=data_sq, location=(0, 200))

    matplotlib.rcParams['axes.edgecolor'] = '#d3d3d3'
    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    fig.set_facecolor((0.106, 0.106, 0.106))
    x, y = update_plot_relu(0, 0)
    ax = fig.add_subplot(111)
    ax.set_ylim([-2, 5])
    ax.set_facecolor((0.106, 0.106, 0.106))
    ax.tick_params(axis='x', colors=(0.827, 0.827, 0.827))
    ax.tick_params(axis='y', colors=(0.827, 0.827, 0.827))
    act_plot, = ax.plot(x, y, color=(0.827, 0.827, 0.827))
    tkcanvas = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    random_seed = 1

    timestamp = time.time()
    interaction_seq_mem = [[a_vals[:], b_vals[:], act[:], gain[:], [random_seed], [int(timestamp)]]]

    torch.manual_seed(random_seed)
    sample_z = torch.randn(cfg.MODEL.sample, cfg.MODEL.latent, device=cfg.MODEL.device)
    sample_np = sample_z.detach().cpu().numpy()
    sample_persistant = copy.deepcopy(sample_np)
    sample_persistant_orig = copy.deepcopy(sample_np)
    img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, [], act, gain, [], random_seed)
    img_save = img_post(img)

    polygon = calc_polygon(sample_np, sample_persistant, magnitude)
    window = draw_poly(polygon, window)

    img = Image.fromarray(img_save)
    bio = io.BytesIO()
    img.save(bio, format= 'PNG')
    imgbytes = bio.getvalue()
    window['-IMAGE-'].Update(data=imgbytes)

    window.bind('<Key-Shift_L>', 'Shift_Down')
    window.bind('<Key-Shift_R>', 'Shift_Down')
    window.bind('<Control-z>', 'UNDO')
    window.bind('<Control-y>', 'REDO')
    shift = False

    window = visibility_logic(window, True)

    state = []
    state_total = []
    state_redo = []
    undo_step = 0
    undo_flag = False
    redo_flag = False
    first_undo_flag = True
    first_redo_flag = True

    slider_flag = False

    while True:
        if start:
            window = draw_graph(window, [ans], layers_act, node_pos, layers2disable, graph_image_persistant)
            window['-PLOT-'].update(plot_name + node_names[1])
            window = visibility_logic(window, False)
            window, update_plot = update_rad(act[ans], window)

            block = False
            window, act_plot, fig = update_viz(a_vals[ans], b_vals[ans], window, act_plot, fig, update_plot)

            layers_group = [ans]
            idx_group = [ans]
            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]
            start = False

        event, values = window.read()

        if event not in ['UNDO', 'REDO']:
            first_undo_flag = True
            first_redo_flag = True

        if event == 'UNDO':
            if first_undo_flag:
                first_undo_flag = False
                undo_step = -2
            else:
                undo_step = -1
            if len(state) == 1:
                continue
            undo_flag = True
            redo_flag = False

            sample_z = state[undo_step][0][0]
            magnitude = state[undo_step][0][1]
            brush_size = state[undo_step][0][2]
            a_vals = state[undo_step][1][0]
            b_vals = state[undo_step][1][1]
            a_min = state[undo_step][1][2]
            a_max = state[undo_step][1][3]
            b_min = state[undo_step][1][4]
            b_max = state[undo_step][1][5]
            act = state[undo_step][1][6]
            gain = state[undo_step][2]
            noise_gen_np = state[undo_step][3]
            noise_gen = state[undo_step][4]
            noise_params = state[undo_step][5]
            layers2disable = state[undo_step][6]
            random_seed = state[undo_step][7]
            idx_group = state[undo_step][8]
            layers_group = state[undo_step][9]

            window["brush_size"].update(brush_size)
            window["magnitude"].update(magnitude)

            values['SEED'] = random_seed
            torch.manual_seed(random_seed)

            window['a_min'].update(a_min)
            window['a_max'].update(a_max)
            window['b_min'].update(b_min)
            window['b_max'].update(b_max)

            a_graph, b_graph = a_vals[layers_group[0]], b_vals[layers_group[0]]

            a_graph = (a_graph - a_min) / (a_max - a_min)
            b_graph = (b_graph - b_min) / (b_max - b_min)
            a_graph *= 200
            b_graph *= 200

            array_sq = cv2.circle(array_sq_persistant.copy(), (int(a_graph), 200 - int(b_graph)), 2, (255,0,0), 2)
            data_sq = array_to_data(array_sq)
            window["ab_graph"].draw_image(data=data_sq, location=(0, 200))

            window = draw_graph(window, layers_group, layers_act, node_pos, layers2disable, graph_image_persistant)

            window, update_plot = update_rad(act[layers_group[0]], window)
            if act[layers_group[0]] == -1:
                window = visibility_logic(window, False)
            else:
                window = visibility_logic(window, True)

            block = False
            if layers_group[-1] in layers2disable:
                block = True
            window, act_plot, fig = update_viz(a_vals[layers_group[0]], b_vals[layers_group[0]], window, act_plot, fig, update_plot, block)

            sample_np = sample_z.detach().cpu().numpy()
            polygon = calc_polygon(sample_np, sample_persistant, magnitude)
            window = draw_poly(polygon, window)

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)
            time.sleep(0.1)

            state_mem = copy.deepcopy(state[-1])
            state.pop()
            state_redo += [state_mem]

        if event == 'REDO':
            if len(state_redo) < 1:
                continue
            if len(state_redo) == 1:
                redo_step = 0
            else:
                redo_step = -1
            undo_flag = False
            redo_flag = True

            sample_z = state[undo_step][0][0]
            magnitude = state[undo_step][0][1]
            brush_size = state[undo_step][0][2]
            a_vals = state[undo_step][1][0]
            b_vals = state[undo_step][1][1]
            a_min = state[undo_step][1][2]
            a_max = state[undo_step][1][3]
            b_min = state[undo_step][1][4]
            b_max = state[undo_step][1][5]
            act = state[undo_step][1][6]
            gain = state[undo_step][2]
            noise_gen_np = state[undo_step][3]
            noise_gen = state[undo_step][4]
            noise_params = state[undo_step][5]
            layers2disable = state[undo_step][6]
            random_seed = state[undo_step][7]
            idx_group = state[undo_step][8]
            layers_group = state[undo_step][9]

            window["brush_size"].update(brush_size)
            window["magnitude"].update(magnitude)

            values['SEED'] = random_seed
            torch.manual_seed(random_seed)

            window['a_min'].update(a_min)
            window['a_max'].update(a_max)
            window['b_min'].update(b_min)
            window['b_max'].update(b_max)

            a_graph, b_graph = a_vals[layers_group[0]], b_vals[layers_group[0]]

            a_graph = (a_graph - a_min) / (a_max - a_min)
            b_graph = (b_graph - b_min) / (b_max - b_min)
            a_graph *= 200
            b_graph *= 200

            array_sq = cv2.circle(array_sq_persistant.copy(), (int(a_graph), 200 - int(b_graph)), 2, (255,0,0), 2)
            data_sq = array_to_data(array_sq)
            window["ab_graph"].draw_image(data=data_sq, location=(0, 200))

            window = draw_graph(window, layers_group, layers_act, node_pos, layers2disable, graph_image_persistant)

            window, update_plot = update_rad(act[layers_group[0]], window)
            if act[layers_group[0]] == -1:
                window = visibility_logic(window, False)
            else:
                window = visibility_logic(window, True)
            block = False
            if layers_group[-1] in layers2disable:
                block = True
            window, act_plot, fig = update_viz(a_vals[layers_group[0]], b_vals[layers_group[0]], window, act_plot, fig, update_plot, block)

            sample_np = sample_z.detach().cpu().numpy()
            polygon = calc_polygon(sample_np, sample_persistant, magnitude)
            window = draw_poly(polygon, window)

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)
            time.sleep(0.1)

            state_redo_mem = copy.deepcopy(state_redo[redo_step])
            state_redo.pop()
            state += [state_redo_mem]
        
        if event == "disable_node":
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            for l in layers_group:
                if l != -1:
                    layers2disable += [l]
            layers2disable = list(set(layers2disable))

            window = draw_graph(window, layers_group, layers_act, node_pos, layers2disable, graph_image_persistant)

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]

            block = False
            if layers_group[-1] in layers2disable:
                block = True
            window, act_plot, fig = update_viz(a_vals[layers_group[0]], b_vals[layers_group[0]], window, act_plot, fig, update_plot, block)

        if event == "restore_node":
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            for l in layers_group:
                if l != -1 and l in layers2disable:
                    layers2disable.remove(l)

            window = draw_graph(window, layers_group, layers_act, node_pos, layers2disable, graph_image_persistant)

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]


        if event == "restore_all_nodes":
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            layers2disable = []
            layers_group = [1]
            idx_group = [1]
            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)

            window = draw_graph(window, layers_group, layers_act, node_pos, layers2disable, graph_image_persistant)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]

        if event == "magnitude":
            slider_flag = True
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            magnitude = float(values["magnitude"])
            sample_np = sample_z.detach().cpu().numpy()
            polygon = calc_polygon(sample_np, sample_persistant, magnitude)
            window = draw_poly(polygon, window)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np, 
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]

        if event == "brush_size":
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            brush_size = int(values["brush_size"])
            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np, 
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]
        if event == "reset-walk":
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            brush_size = 4
            magnitude = 1.0
            window["brush_size"].update(brush_size)
            window["magnitude"].update(magnitude)
            sample_np = copy.deepcopy(sample_persistant_orig)
            sample_z = torch.from_numpy(sample_np).cuda()
            sample_persistant = copy.deepcopy(sample_persistant_orig)
            polygon = calc_polygon(sample_np, sample_persistant, magnitude)
            window = draw_poly(polygon, window)

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]

        if event == "walk_graph":
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            a_graph, b_graph = values[event]
            b_graph = 200 - b_graph
            x_graph = a_graph - 100
            y_graph = b_graph - 100
            r_coord = math.sqrt( x_graph**2 + y_graph**2)
            angle = math.degrees(math.acos(x_graph/r_coord))

            # sample_np = 50 * (sample_np - np.min(sample_np)) / (np.max(sample_np) - np.min(sample_np))
            r = (r_coord * (np.max(sample_np) - np.min(sample_np)) / (50*magnitude) ) + np.min(sample_np) 

            if y_graph < 0:
                angle = 360 - angle
            ind_angle = np.uint16((angle * 512)/360)

            close_ind = list(range(-brush_size, brush_size))

            for i in close_ind:
                ii = ind_angle + i
                if ii >= 512:
                    ii = i
                elif ii < 0:
                    ii = 511 + i
                polygon[ii] = [a_graph, b_graph]
                sample_z[:, ii] = r

            array_walk = np.zeros((200,200,3), dtype = np.uint8)
            array_walk = cv2.polylines(array_walk, [polygon], True, (211, 211, 211), 1)
            array_walk_persistant = array_walk.copy()
            data_walk = array_to_data(array_walk)
            window["walk_graph"].draw_image(data=data_walk, location=(0, 200))

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]


        if event == "ab_graph":
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            a_graph, b_graph = values[event]

            array_sq = cv2.circle(array_sq_persistant.copy(), (int(a_graph), 200 - int(b_graph)), 2, (255,0,0), 2)
            data_sq = array_to_data(array_sq)
            window["ab_graph"].draw_image(data=data_sq, location=(0, 200))

            a_graph /= 200
            b_graph /= 200
            a_graph = ( (a_max - a_min) * a_graph) + a_min
            b_graph = ( (b_max - b_min) * b_graph) + b_min

            window['x_text'].update('X: ' + str(a_graph))
            window['y_text'].update('Y: ' + str(b_graph))

            for i in layers_group:
                a_vals[i] = a_graph
                b_vals[i] = b_graph

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)
            time.sleep(0.1)

            window, act_plot, fig = update_viz(a_vals[layers_group[-1]], b_vals[layers_group[-1]], window, act_plot, fig, update_plot)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]


        if event == 'Shift_Down':
            shift = True
        if event == 'Shift_L:50':
            shift = False

        if event == "viz_graph":
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            x, y = values[event]
            ans = find_bbox(x, y, node_pos)

            a_graph, b_graph = a_vals[ans], b_vals[ans]
            a_graph = (a_graph - a_min) / (a_max - a_min)
            b_graph = (b_graph - b_min) / (b_max - b_min)
            a_graph *= 200
            b_graph *= 200
            array_sq = cv2.circle(array_sq_persistant.copy(), (int(a_graph), 200 - int(b_graph)), 2, (255,0,0), 2)
            data_sq = array_to_data(array_sq)
            window["ab_graph"].draw_image(data=data_sq, location=(0, 200))

            if shift == True:
                if ans not in [-1,0,9]:
                    layers_group += [ans]
                    layers_group = list(set(layers_group))
            if shift == False:
                if ans not in [-1,0,9]:
                    layers_group = [ans]

            
            for l in layers_group:
                if l in layers_act:
                    idx_group += [l]
            idx_group = list(set(idx_group))

            if ans not in [-1,0,9]:
                window, update_plot = update_rad(act[ans], window)
                window = draw_graph(window, layers_group, layers_act, node_pos, layers2disable, graph_image_persistant)

                block = False
                if ans in layers2disable:
                    block = True
                print(block)
                window, act_plot, fig = update_viz(a_vals[ans], b_vals[ans], window, act_plot, fig, update_plot, block)

                if act[ans] == -1 :
                    window = visibility_logic2(window, True)
                    window = visibility_logic(window, False)
                elif check_rgb(layers_group, node_names):
                    window = visibility_logic2(window, False)
                else:
                    window = visibility_logic2(window, True)

                if shift == False:
                    state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                                       noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
                    state_total += [copy.deepcopy(state[-1])]

            if ans in layers_act and len(layers_group) == 1:
                window['-PLOT-'].update(plot_name + node_names[ans])
            if len(layers_group) > 1:
                current_name = plot_name + "multiple"

        if event == 'Shift_L:50':
            if len(layers_group) > 0:
                state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                                   noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
                state_total += [copy.deepcopy(state[-1])]
            
        if event == '-GEN-':
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            random_seed = int(values['SEED'])
            torch.manual_seed(random_seed)
            sample_z = torch.randn(1, cfg.MODEL.latent, device=cfg.MODEL.device)
            sample_np = sample_z.detach().cpu().numpy()
            sample_persistant = copy.deepcopy(sample_np)
            sample_persistant_orig = copy.deepcopy(sample_np)
            polygon = calc_polygon(sample_np, sample_persistant, magnitude)
            window = draw_poly(polygon, window)

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)
            time.sleep(0.1)

            block = False
            if layers_group[-1] in layers2disable:
                block = True
            window, act_plot, fig = update_viz(a_vals[layers_group[-1]], b_vals[layers_group[-1]], window, act_plot, fig, update_plot, block)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]
            
        if event == '-RESET-':
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            for l in layers_group:
                act[l] = -1
                a_vals[l] = 0
                b_vals[l] = 0

            window = visibility_logic(window, False)

            update_plot = update_plot_relu
            block = False
            if layers_group[-1] in layers2disable:
                block = True
            window, act_plot, fig = update_viz(a_vals[layers_group[-1]], b_vals[layers_group[-1]], window, act_plot, fig, update_plot, block)
            window['relu'].update(value=True)

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]

        if event == '-RESETALL-':
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            update_plot = update_plot_relu
            for i in range(len(act)):
                act[i] = -1
                a_vals[i] = 0
                b_vals[i] = 0

            window = visibility_logic(window, False)

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)
            window, act_plot, fig = update_viz(0, 0, window, act_plot, fig, update_plot)
            window['relu'].update(value=True)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]


        if event in act_names:
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            if event == act_names[0]:
                update_plot = update_plot_relu
            elif event == act_names[1]:
                update_plot = update_plot_sin
            elif event == act_names[2]:
                update_plot = update_plot_cos
            elif event == act_names[3]:
                update_plot = update_plot_ren
            elif event == act_names[4]:
                update_plot = update_plot_shi

            act_id = act_names.index(event) - 1

            for i in layers_group:
                act[i] = act_id
            block = False
            if layers_group[-1] in layers2disable:
                block = True
            window, act_plot, fig = update_viz(a_vals[layers_group[-1]], b_vals[layers_group[-1]], window, act_plot, fig, update_plot, block)

            if act_id == -1:
                window = visibility_logic(window, False)
            else:
                window = visibility_logic(window, True)

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]
            
        if event == 'set':
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            a_min = float(values['a_min'])
            a_max = float(values['a_max'])
            b_min = float(values['b_min'])
            b_max = float(values['b_max'])

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]
            
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-SAVE-":
            timestamp = time.time()
            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_out = img_post(img)[...,::-1]
            cv2.imwrite('./output/' + str(int(timestamp)) + '.png', img_out)
            # torch.save(noise_gen, './raw/noise/'+str(int(timestamp))+'.pt')
            # interaction_seq = [a_vals, b_vals, act, gain, [random_seed], [int(timestamp)]]

            with open("./output/"+str(int(timestamp))+".csv", 'w') as f:
               writer = csv.writer(f)
               writer.writerows(state_total)

            state_total = []
            # with open("./raw/raw/"+str(int(timestamp))+"_1.csv", 'w') as f:
            #    writer = csv.writer(f, delimiter='\n')
            #    writer.writerows(interaction_seq_mem)

        
        if event in ['-Noise-']: 
            if undo_flag:
                undo_flag = False
                state += [state_mem]
                state_redo = []
            if redo_flag:
                undo_flag = False
                state += [state_redo_mem]
                state_redo = []

            noise_gen_np,noise_gen,noise_params,gain,interaction_seq_mem = noise_window(noise_gen_np, noise_gen, noise_params, gain, interaction_seq_mem)

            # gain = [float(values["noise_set"]) for g in gain]

            img = generate(g_ema, sample_z, mean_latent, noise_gen, a_vals, b_vals, idx_group, act, gain, layers2disable, random_seed)
            img_save = img_post(img)

            img = Image.fromarray(img_save)
            bio = io.BytesIO()
            img.save(bio, format= 'PNG')
            imgbytes = bio.getvalue()
            window['-IMAGE-'].Update(data=imgbytes)

            state = add2state([sample_z, magnitude, brush_size], [a_vals, b_vals, a_min, a_max, b_min, b_max, act], gain, noise_gen_np,
                               noise_gen, noise_params, layers2disable, random_seed, idx_group, layers_group, state)
            state_total += [copy.deepcopy(state[-1])]
        
    window.close()


if __name__ == '__main__':
    import argparse

    import torch
    from model import Generator

    import numpy as np
    import time, math
    import matplotlib.pyplot as plt

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import PySimpleGUI as sg
    import matplotlib
    import os.path
    import io, csv
    import cv2

    import copy
    from PIL import Image

    from config import cfg 

    from utils.act_func import *
    from utils.ui_utils import * 
    from utils.common import * 
    from utils.gui import build, viz_prep, noise_window

    random_seed = 1
    torch.manual_seed(random_seed)

    main()