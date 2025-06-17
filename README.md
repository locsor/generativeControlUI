GUI for StyleGAN2

Demo: https://www.youtube.com/watch?v=qkP9DHLicwM

Currently only compatible with StyleGAN 2 models with following charactersitcs:
- 1024x1024x3 output
- 8 layer MLP
- 512-dimensional latent vector 

Link to model checkpoint: https://drive.google.com/file/d/1I6TI91GZKbho5Uy7XffCQLwAK1v-rUvD/view?usp=sharing

1. Set weight file path in ./config/defaults.py file
2. Run ui.py file

Tested with Cuda 11.0, PyTorch 1.7.1, and Python 3.9.

![UI Image](https://github.com/locsor/generativeControlUI/blob/master/images/UI.png?raw=true)

1. Neural Network architecture visualizer (a).<br>Buttons for enabling/disabling layers (b).
2. Activation functions selector and visualizer (a).<br>Interface to control trainable activation functions (b).
3. Interface for a traversing model's latent space.
4. Output image visualizer, button to enter the noise editor, buttons to reset activation functions, random seed input field. 

Examples:

<img src="https://github.com/locsor/generativeControlUI/blob/master/images/out1.png" width="50%" height="50%">
<img src="https://github.com/locsor/generativeControlUI/blob/master/images/out3.png" width="50%" height="50%">
<img src="https://github.com/locsor/generativeControlUI/blob/master/images/out4.png" width="50%" height="50%">
![Gif1](https://github.com/locsor/generativeControlUI/blob/master/images/gif_1.gif)
![Gif2](https://github.com/locsor/generativeControlUI/blob/master/images/gif_2.gif)


