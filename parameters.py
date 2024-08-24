import torch
from thop import profile
import networks

inputs=torch.randn(1,3,640,192)
encoder = networks.ResnetEncoder(50, True)
# encoder = networks.convnext()
# depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
# depth_decoder = networks.ConvDecoder(encoder.num_ch_enc)


flops , params=profile(encoder,(inputs,))

print("gflops:{:.2f},params:{:.2f}".format(flops/1e9,params/1e6))

