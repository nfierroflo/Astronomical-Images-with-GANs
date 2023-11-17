from torch import nn
from src.utils.tools import make_disc_block

        
class Discriminator(nn.Module):
    def __init__(self, im_chan=3, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            make_disc_block(im_chan,hidden_dim,kernel_size=4,stride=2),
            make_disc_block(hidden_dim,hidden_dim*2),
            make_disc_block(hidden_dim*2,1,final_layer=True)
        )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)