'''
 CoordConv code source :https://github.com/mkocabas/CoordConv-pytorch
 Shuffenet code source :https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2
 Inception block code source :https://github.com/Lornatang/GoogLeNet-PyTorch/blob/master/googlenet_pytorch/model.py

 @inproceedings{ma2018shufflenet, 
            title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},  
            author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},  
            booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
            pages={116--131}, 
            year={2018} 
}
'''






from turtle import forward
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torchvision import models
class base_Conv(nn.Module):

    def __init__(self, in_channels, out_channels,ksize,mid=1,padding=1):
        super(base_Conv, self).__init__()
        self.ksize=ksize
        self.conv1 = nn.Conv2d(in_channels, mid,kernel_size=1,stride=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act1=nn.SiLU()
        self.conv2 = nn.Conv2d(mid, out_channels,kernel_size=ksize,stride=2,padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2=nn.SiLU()


        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=ksize,stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act=nn.SiLU()
    def forward(self, x):
        if self.ksize!=1:
            x = self.conv1(x)
            x = self.bn1(x)
            x=self.act1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x=self.act2(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x=self.act(x)
        return x
class Inception_block(nn.Module):
    def __init__(self,in_channel):
        super(Inception_block,self).__init__()
      
        self.branch_1_1=base_Conv(in_channel,10,1)
        self.branch_3_3=base_Conv(in_channel,7,3,6)
        self.branch_5_5=base_Conv(in_channel,4,5,6,padding=2)
        self.branch_pool=nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            base_Conv(in_channel, 3, 1)
        )
    def _forward(self,x):
        branch_1_1=self.branch_1_1(x)
        branch_3_3=self.branch_3_3(x)
        branch_5_5=self.branch_5_5(x)
        branch_pool=self.branch_pool(x)
        outputs=[branch_1_1,branch_3_3,branch_5_5, branch_pool]
     
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)






class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class ShuffleNetV2_ver3(nn.Module):

    def __init__(self, in_channels ,nclass=136, with_r=True, extra_channel=0,**kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        self.shufflenet=ShuffleNetV2(extra_channel=extra_channel,n_class=nclass)
        in_size = in_channels+2+extra_channel
        if with_r:
            in_size += 1
        

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.shufflenet(ret)
        return ret

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.SiLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.SiLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, n_class=1000, model_size='1.7x',extra_channel=0):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size =='1.7x':
            self.stage_out_channels = [-1, 24, 192, 396, 764, 1172]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(6+extra_channel, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.SiLU(inplace=True),
        )
        self.Inception_block=Inception_block(6+extra_channel)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel
                
        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.SiLU(inplace=True)
        )
        self.globalpool = nn.AvgPool2d(7)
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        #X = self.first_conv(x)
  
        x=self.Inception_block(x)
        
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size == '2.0x':
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
