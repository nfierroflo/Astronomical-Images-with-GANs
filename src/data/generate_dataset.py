import sys
sys.path.insert(0,"/home/nfierroflo/Documents/Redes Neuronales/Astronomical-Images-with-GANs")

from src.Generator import Generator
from src.utils.tools import * 
import torch
import pickle
from tqdm import tqdm


def generate_dataset(model_path,size):

    device = 'cuda'
    z_dim = 64
    #create dataset dictionary
    dataset = {'images':[],'labels':[]}
    for i in tqdm(range(int(size))):

        num_img=1
        gen= Generator(z_dim).to(device)
        gen.load_state_dict(torch.load(model_path))
    
        fake_noise = get_noise(num_img, z_dim, device=device)
        fake_sample = gen(fake_noise)
        
        dataset['images'].append(fake_sample.cpu().detach().numpy()[0])
        dataset['labels'].append(1.0)
    
    #save the dataset as a pickle file
    with open(f'generated_dataset_{size}_model{model_path}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    model_path = sys.argv[1]
    size = sys.argv[2]
    generate_dataset(model_path,size)