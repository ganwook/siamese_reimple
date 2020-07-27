import random
import os
import argparse
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader

random.seed(10)


def split_eval(back_dir, eval_dir):

    idxs_eval = random.sample(range(20), 8)
    idxs_test = random.sample(idxs_eval, 4)
    idxs_valid = list(set(idxs_eval) - set(idxs_test))
    idxs_back = list(set(range(20)) - set(idxs_eval))

    idxs_test_alp = random.sample(range(20), 10)
    idxs_valid_alp = list(set(range(20)) - set(idxs_test_alp))

    idxs_drawer = {'back'  : idxs_back,
                   'test'  : idxs_test,
                   'valid' : idxs_valid,
                  }
    idxs_alpha =  {'test'  : idxs_test_alp,
                   'valid' : idxs_valid_alp,
                  }
    dic_idxs_eval = {'drawer'   : idxs_drawer,
                     'alphabet' : idxs_alpha}

    return dic_idxs_eval

def sample_subtree(path):
    current = random.choice(os.listdir(path))
    subtree = random.choice(os.listdir(os.path.join(path, current)))
    subtreePath = os.path.join(path, current, subtree)

    return current, subtree, subtreePath

class VerificationDataset(Dataset):
    """ Dataset for the Verification Task """
    def __init__(self, dir_back, n_samples, idxs_back):
        self.root_dir = dir_back
        self.n_samples = n_samples
        self.idxs_back = idxs_back

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Generate samples according to the label. 
        label = idx % 2
        
        # 'Same'
        if label == 1:
            alpha, char, charPath = sample_subtree(self.root_dir)
            idxs_drawers = random.sample(self.idxs_back, k=2)
            drawers = [os.listdir(charPath)[idx] for idx in idxs_drawers]
            img_names = [os.path.join(charPath, drawers[i]) for i in range(2)]
        # 'Different'
        else:
            alpha1, char1, charPath1 = sample_subtree(self.root_dir)
            
            diff = False
            while not diff:    
                alpha2, char2, charPath2 = sample_subtree(self.root_dir)
                diff = True if charPath1 != charPath2 else False
            
            charPaths = [charPath1, charPath2]
            idxs_drawer = [os.listdir(charPaths[i])[random.choice(self.idxs_back)] for i in range(2)]
            img_names = [os.path.join(charPaths[i], idxs_drawer[i]) for i in range(2)]
            
        img1 = io.imread(img_names[0])
        img2 = io.imread(img_names[1])

        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))


if __name__ == '__main__':
    
    random.seed(10)

    parser = argparse.ArgumentParser()

    parser.add_argument('--back_dir', default='./data/background')
    parser.add_argument('--eval_dir', default='./data/evaluation')

    args = parser.parse_args()

    assert len(os.listdir(args.back_dir))==30
    assert len(os.listdir(args.eval_dir))==20
    
    print(dic_idxs_eval)

    
    