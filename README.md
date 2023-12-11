GUI for StyleGAN2

Currently only compatible with StyleGAN 2 models with following charactersitcs:
- 1024x1024x3 output
- 8 layer MLP
- 512-dimensional latent vector 

Link to model checkpoint: https://drive.google.com/file/d/1I6TI91GZKbho5Uy7XffCQLwAK1v-rUvD/view?usp=sharing

1. Set weight file path in ./config/defaults.py file
2. Run ui.py file

Tested with Cuda 11.0 and PyTorch 1.7.1

![alt text](https://github.com/locsor/generativeControlUI/blob/master/images/17_1_line.png?raw=true)
1a. Neural Network architecture visualizer.
1b. Buttons for enabling/disabling layers.
2a. Activation functions selector and visualizer.
2b. Interface to control trainable activation functions.
3. Interface for a traversing model's latent space.
4. Output image visualizer, button to enter the noise editor, buttons to reset activation functions, random seed input field. 

Examples:
![alt text](https://github.com/locsor/generativeControlUI/blob/master/images/out1.png?raw=true)
![alt text](https://github.com/locsor/generativeControlUI/blob/master/images/out3.png?raw=true)
![alt text](https://github.com/locsor/generativeControlUI/blob/master/images/out4.png?raw=true)
