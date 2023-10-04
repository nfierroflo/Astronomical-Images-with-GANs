from torch import randn
from torch import nn

def make_gen_block(input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
    if not final_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
            nn.BatchNorm2d(output_channels),nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
            nn.Tanh()
        )

def make_disc_block(input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_channels),nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride)

            )
def get_noise(n_samples, z_dim, device='cpu'):
    return randn(n_samples, z_dim, device=device)