import torch
import numpy as np
import copy
import cv2
from io import BytesIO
from PIL import Image

def array_to_data(array):
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data


def add2state(sample_etc, act_etc, gain, noise_gen_np, noise_gen, noise_params, layers2disable, random_seed, idxs, layers, state):
    state += [[copy.deepcopy(sample_etc), copy.deepcopy(act_etc), copy.deepcopy(gain), copy.deepcopy(noise_gen_np),
               copy.deepcopy(noise_gen), copy.deepcopy(noise_params), copy.deepcopy(layers2disable), random_seed, copy.deepcopy(idxs), copy.deepcopy(layers)]]
    return state

def img_post(img):
    img = torch.clamp(img, min=-1, max=1)
    img = img * 0.5 + 0.5
    img = np.uint8(img[0].cpu().numpy().transpose(1, 2, 0) * 255)
    img = cv2.resize(img, (512,512))
    return img

def img_post_save(img, name):
    img = torch.clamp(img, min=-1, max=1)
    img = img * 0.5 + 0.5
    img = np.uint8(img[0].cpu().numpy().transpose(1, 2, 0) * 255)
    img = cv2.resize(img, (512,512))
    cv2.imwrite('./output/' + str(name) + '.png', img[:,:,::-1])
