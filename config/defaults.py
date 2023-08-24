from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.device = "cuda"
_C.MODEL.size = 1024
_C.MODEL.truncation = 1
_C.MODEL.truncation_mean = 4096
_C.MODEL.ckpt = "./weights/stylegan2-ffhq-config-f.pt"
_C.MODEL.sample = 1
_C.MODEL.channel_multiplier = 2
_C.MODEL.latent = 512
_C.MODEL.n_mlp = 8

_C.MODEL.layer_markings =[-1, 0, 1, 2, 3, 4, 5, 6, 7, -1, 
						  8, -1, 9, 10, -1, 11, 12, -1,
						  13, 14, -1, 15, 16, -1, 17, 18,
						  -1, 19, 20, -1, 21, 22, -1, 23,
						  24, -1]

