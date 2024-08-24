from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *

class Resblock(nn.Module):
    def __init__(self,in_c,out_c):
        super(Resblock, self).__init__()
        self.conv3x3=Conv3x3(in_c,out_c)
        self.relu=nn.ReLU()
    def forward(self,x):
        out=x
        x=self.conv3x3(x)
        x=self.relu(x)
        x = self.conv3x3(x)
        x = self.relu(x)

        output=out+x

        return output


class Enh(nn.Module):
    def __init__(self,in_c,out_c):
        super(Enh, self).__init__()
        self.conv1x1 = nn.Conv2d(out_c, out_c, 1)
        self.resblock=Resblock(out_c,out_c)
        self.convb = ConvBlock(in_c, out_c)

    def forward(self,input):
        x,y=input
        bs, c, h, w = y.shape
        Add=x + y
        Att=self.resblock(y)
        Att=self.conv1x1(Att)
        Att=F.softmax(Att.view(bs,c,-1),dim=-1)
        Mul=Att * Add.view(bs,c,-1)
        Mul=Mul.view(bs, c, h, w)
        Add=Add.view(bs, c, h, w)
        z=self.resblock(Mul)
        z=[z] + [Add]
        z=torch.cat(z,1)
        z = self.convb(z)
        z=upsample(z)


        return z




class ConvDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(ConvDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_enc_0=np.array([256,256,512,1024])
        self.num_ch_dec = np.array([128, 128,256, 512])

        # decoder
        self.convs = OrderedDict()
        self.dec = nn.ModuleList()
        for i in range(4):
            dec=nn.Sequential(
                              *[Enh(self.num_ch_enc_0[i],self.num_ch_dec[i])]
                )
            self.dec.append(dec)

            self.convs[("conv1x1", i,0)]=ODConv2d(self.num_ch_enc[i], self.num_ch_enc[i], 1)
            self.convs[("conv3x3", i, 1)] = ConvBlock(self.num_ch_enc[i], self.num_ch_dec[i])


        for s in self.scales:
            self.convs[("dispconv", s)] = Head(self.num_ch_dec[s])

        self.decoder = nn.ModuleList(list(self.convs.values()))


    def forward(self, input_features):
        self.outputs = {}
        f3 = self.convs[("conv1x1", 3,0)](input_features[3])
        f2 = self.convs[("conv1x1", 2,0)](input_features[2])
        f1 = self.convs[("conv1x1", 1,0)](input_features[1])
        f0 = self.convs[("conv1x1", 0,0)](input_features[0])



        x=self.convs[("conv3x3", 3,1)](f3)
        x = upsample(x)
        x=self.dec[3]((x,f2))
        self.outputs[("disp", 3)]= self.convs[("dispconv", 3)](x)


        x=self.convs[("conv3x3", 2,1)](x)
        x=self.dec[2]((x,f1))
        self.outputs[("disp", 2)]= self.convs[("dispconv", 2)](x)



        x=self.convs[("conv3x3", 1,1)](x)
        x=self.dec[1]((x,f0))
        self.outputs[("disp", 1)] = self.convs[("dispconv", 1)](x)



        x=self.convs[("conv3x3", 0,1)](x)
        f0_up=upsample(f0)
        x=self.dec[0]((x,f0_up))
        self.outputs[("disp", 0)]= self.convs[("dispconv", 0)](x)









        return self.outputs
