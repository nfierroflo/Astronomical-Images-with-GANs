
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from scipy.stats import entropy

class Logger(object):
    def __init__(self, log_dir, **kwargs):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir, **kwargs)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images.
        Args::images: numpy of shape (Batch x C x H x W) in the range [-1.0, 1.0]
        """

        if type(images) == tuple or type(images) == list:
            images = np.array(images)

        if len(images.shape) == 3:
            images = images[:,:,:,None]
            self.writer.add_images("{}".format(tag), images, global_step=step, dataformats="NHWC")
            return    
        
        self.writer.add_images("{}".format(tag), images, global_step=step, dataformats="NCHW")

            
    def histo_summary(self, tag, values, step, bins="auto"):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram('{}'.format(tag), values, bins=bins, global_step=step)




def get_inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """
        Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

