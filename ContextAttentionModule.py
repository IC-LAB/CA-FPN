import torch
import torch.nn as nn

"""
context attentnion module
"""
class DilatedContextAttentionModule(nn.Module):
    def __init__(self, mode='dot_product', channels=256):
        super(DilatedContextAttentionModule, self).__init__()
        self.in_channels=channels
        self.inter_channels=channels
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0)
        
        self.bn = nn.BatchNorm2d(self.in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        if mode=='dot_product':
            self.operation_function = self._dot_product
        else:
            print("not implement")
 
    def _dot_product(self, xi, xj):
        batch_size = xi.size(0)
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(xj).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(xi).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(xj).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *xi.size()[2:])
        W_y = self.W(y)
        z = W_y + xi
        z = self.bn(z)
        return z

    def forward(self, xi, xj):
        '''
        :param xi, xj: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        output = self.operation_function(xi, xj)
        return output
 
class ContextAttentionModule(nn.Module):

    def __init__(self, padding_dilation, input_channels):
        super(ContextAttentionModule, self).__init__()
        self.dilation = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=padding_dilation, dilation=padding_dilation,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dcam = DilatedContextAttentionModule(mode='dot_product', channels=input_channels)
        self.wr = nn.Conv2d(input_channels*2, input_channels, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, pn):
        pn_dilated = self.dilation(pn)
        pn_dilated = self.relu(pn_dilated)
        pn_dcam = self.dcam(pn_dilated, pn)
        concat = torch.cat((pn_dcam, pn),1)
        out = self.wr(concat)
        return out

if __name__ == '__main__':
    img = torch.randn(2, 3, 16, 16) #(b, c, h, w)
    print("input size:", img.size())
    cam = ContextAttentionModule(padding_dilation=2, input_channels=3)
    out = cam(img)
    print(out)
    print("output size", out.size())