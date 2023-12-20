import numpy as np
from scipy.stats import multivariate_normal
import pickle as pk
from tqdm import tqdm


def create_SN(l=28,min_cov=1,max_cov=3):

    # Set mean and covariance for the Gaussian distribution
    mean = [l//2, l//2]
    #generate a numbre between 
    cov=np.random.uniform(min_cov,max_cov)
    covariance = [[cov, 0], [0, cov]]  # Adjust covariance matrix as needed

    # Create a grid
    x, y = np.meshgrid(np.arange(l), np.arange(l))

    # Create a multivariate normal distribution
    gaussian = multivariate_normal(mean=mean, cov=covariance)

    # Evaluate the Gaussian at each point in the grid
    Sn = gaussian.pdf(np.dstack((x, y)))

    # Normalize
    Sn = Sn / Sn.max()

    return Sn

def load_references(data_dir='data/',data_name="stamp_dataset_28.pkl"):
    #Carga de datos
    with open(data_dir + data_name, "rb") as f:
        data = pk.load(f)

    Train_dict = data['Train']

    train_images = Train_dict['images']

    references=train_images[:,:,:,2]

    return references

def generate_SN_dataset(data_dir,data_name,size,save_path='data/Handcrafted_SN_dataset.pkl'):
    #create dataset dictionary
    dataset = {'images':[],'labels':[]}
    references=load_references(data_dir,data_name)
    for i in tqdm(range(int(size))):
        #generate a random SN
        Sn=create_SN()
        #generate a random reference
        ref=references[np.random.randint(0,references.shape[0])]
        #add the SN to the reference
        Sn_ref=ref+Sn
        #normalize the SN
        
        during = Sn_ref / Sn_ref.max()
        
        difference=Sn

        image=np.array([difference,during,ref])
        #add the SN to the dataset
        dataset['images'].append(image)
        dataset['labels'].append(1.0)

    
    #save the dataset as a pickle file
    with open(save_path, 'wb') as f:
        pk.dump(dataset, f)

if __name__ == "__main__":
    data_dir='data/'
    data_name="stamp_dataset_28.pkl"
    size=50000
    save_path='data/Handcrafted_SN_dataset.pkl'
    generate_SN_dataset(data_dir,data_name,size,save_path)






