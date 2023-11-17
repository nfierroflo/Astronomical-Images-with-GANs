from torch import nn
from src.utils.tools import make_gen_block

    
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            make_gen_block(self.z_dim,hidden_dim*4,kernel_size=3,stride=2),
            make_gen_block(hidden_dim*4,hidden_dim*2,kernel_size=4,stride=1),
            make_gen_block(hidden_dim*2,hidden_dim,kernel_size=3,stride=2),
            make_gen_block(hidden_dim,im_chan,kernel_size=4,stride=2,final_layer=True)
        )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)