import random
import os
import argparse
import torch
import numpy as np
import copy
from skimage import io
from torch.utils.data import Dataset
from PIL import Image

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
        self.datas = self.loadtoMem()

    def loadtoMem(self):
        print(f'begin loading background dataset to the memory')
        datas = {}

        for idx in range(self.n_samples):
            label = idx % 2
            # 'Same' data pair
            if label == 1:
                alpha, char, charPath = sample_subtree(self.root_dir)
                idxs_drawers = random.sample(self.idxs_back, k=2)
                drawers = [os.listdir(charPath)[idx] for idx in idxs_drawers]
                img_names = [os.path.join(charPath, drawers[i]) for i in range(2)]
            # 'Different' data pair
            else:
                alpha1, char1, charPath1 = sample_subtree(self.root_dir)
                
                diff = False
                while not diff:    
                    alpha2, char2, charPath2 = sample_subtree(self.root_dir)
                    diff = True if charPath1 != charPath2 else False
                
                charPaths = [charPath1, charPath2]
                idxs_drawer = [os.listdir(charPaths[i])[random.choice(self.idxs_back)] for i in range(2)]
                img_names = [os.path.join(charPaths[i], idxs_drawer[i]) for i in range(2)]
            
            datas.update({idx : {'imgs'  : img_names,
                                 'label' : label      }})

        print(f'{idx+1} samples are generated.')
        return datas

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """ Generate samples according to the label. """
        img_names = self.datas[idx]['imgs']
        label = self.datas[idx]['label']
        imgs = [Image.open(img_names[i]) for i in range(2)]

        return imgs[0], imgs[1], torch.from_numpy(np.array([label], dtype=np.float32))


class OneshotDataset(Dataset):
    """ Dataset for One-shot image recognition 
    
    args.
        - phase = 'valid' / 'test'
    """
    def __init__(self, dir_eval, n_ways, idxs_eval, phase):
        self.root_dir = dir_eval
        self.n_ways = n_ways
        self.idxs_drawer = idxs_eval['drawer'][phase]
        self.idxs_alpha = idxs_eval['alphabet'][phase]
        self.phase = phase
        self.datas = self.loadtoMem()
    
    def loadtoMem(self):
        """ 
        the pair of drawers => [0, 1] : [2, 3] = input_image : candidates = pair : (pair + 2)
        """
        print(f"begin loading {self.phase} dataset to the memory")
        datas = {} ; idx = 0 
        nms_alpha = os.listdir(self.root_dir)

        for idx_alpha in self.idxs_alpha:
            alphaPath = os.path.join(self.root_dir, nms_alpha[idx_alpha])
            lst_chars = os.listdir(alphaPath)
            
            if len(lst_chars) > 20:
                lst_chars = random.sample(lst_chars, 20)
            elif len(lst_chars) < 20:
                print(alphaPath)
                raise AssertionError('the number of characters should be greater than 20')
            
            # get path and id for all of the selected characters
            dic_alpha = {}
            for l, char in enumerate(lst_chars):
                charPath = os.path.join(alphaPath, char)
                dic_alpha[l] = {'path': charPath,
                                'id'  : os.listdir(charPath)[0].split('_')[0] + '_'}
            
            # Generate single trial from dic_alpha and drawer indeces.
            trial = {}
            for pair in range(2):
                nm_input_drawer = self.idxs_drawer[pair] + 1
                nm_cand_drawer = self.idxs_drawer[pair + 2] + 1
                trial['ways'] = {l : os.path.join(dic['path'], dic['id'] + '{:02}'.format(nm_cand_drawer) + '.png') for (l, dic) in dic_alpha.items()}
                
                for (label, dic) in dic_alpha.items():
                    trial['label'] = label
                    trial['input'] = os.path.join(dic['path'], dic['id'] + '{:02}'.format(nm_input_drawer) + '.png')
                    datas[idx] = copy.deepcopy(trial)
                    idx += 1
                    
        print(f'{idx} samples are generated.')
        return datas

    def __len__(self):
        return 2 * len(self.idxs_alpha) * self.n_ways

    def __getitem__(self, idx):
        """ Generate the one-shot learning samples for evaluation """
        dirs_trial = self.datas[idx]
        input_img = Image.open(dirs_trial['input'])
        ways_img = {l : Image.open(path) for (l, path) in dirs_trial['ways'].items()}
        label = torch.from_numpy(np.array([dirs_trial['label']], dtype=np.float32)) 

        return input_img, ways_img, label



if __name__ == '__main__':
    
    random.seed(10)

    parser = argparse.ArgumentParser()

    parser.add_argument('--back_dir', default='./data/background')
    parser.add_argument('--eval_dir', default='./data/evaluation')

    args = parser.parse_args()

    assert len(os.listdir(args.back_dir))==30
    assert len(os.listdir(args.eval_dir))==20

    dic_idxs_eval = split_eval(args.back_dir, args.eval_dir)
    
    print(dic_idxs_eval)

    
    