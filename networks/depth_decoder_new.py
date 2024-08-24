from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = np.array([128 , 128 , 256 , 512 , 1024])
        self.num_ch_dec = np.array([64 , 128 , 128 , 256 , 512])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]  #20x20x768
        x = self.convs[("upconv", 4, 0)](x)  #768~384
        x = [upsample(x)]   #40
        x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 4, 1)](x) #1024-512 40x40


        x = self.convs[("upconv", 3, 0)](x)  #384~192
        x = [upsample(x)]
        x += [input_features[1]] #80
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 3, 1)](x) #512-256 80x80
        self.outputs[("disp", 3)] = self.sigmoid(self.convs[("dispconv", 3)](x))


        x = self.convs[("upconv", 2, 0)](x) #192~96
        x = [upsample(x)]
        x += [input_features[0]] #160
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x) #256-128 160x160
        self.outputs[("disp", 2)] = self.sigmoid(self.convs[("dispconv", 2)](x))

        x = self.convs[("upconv", 1, 0)](x) #96~96
        x = [upsample(x)]
        input_features[0] = upsample(input_features[0])
        x += [input_features[0]] # 320
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x) #256-128 320x320
        self.outputs[("disp", 1)] = self.sigmoid(self.convs[("dispconv", 1)](x))


        x = self.convs[("upconv", 0, 0)](x) #128-64
        x = [upsample(x)]  # 640
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 0, 1)](x) #64-64 640x640
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](x))




        return self.outputs
