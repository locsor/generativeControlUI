import math
import numpy as np

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
        ys[i] = min(t, 1 * (a+1))
        
    return xs, ys

def update_plot_shi(a, b):
    xs = np.linspace(-5, 5, num=1000)
    ys = np.zeros(1000)
    for i, x in enumerate(xs):
        # ys[i] =  a*x**3 + b*x**2 + 0.5*x
        ys[i] = a * np.maximum(x, 0) + b
        
    return xs, ys

