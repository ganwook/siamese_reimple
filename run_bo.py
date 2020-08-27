import os
import logging
import argparse
import random
import torch

from torchvision import transforms
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from preprocess import VerificationDataset, split_eval, OneshotDataset
from model import SiameseNN
from run_siamese import save_model_and_result, oneshot_eval, train
from config import *

logger = logging.getLogger(__name__)

def train_and_test(args, hparams):

    """ Load the datasets, model, and configs """
    logger.info("***** Loading the datasets, model, and configs *****")

    dic_idxs_eval = split_eval(args.back_dir, args.eval_dir)

    BackgroundDataset = VerificationDataset(args.back_dir, n_samples = args.n_train, 
                                            idxs_back = dic_idxs_eval['drawer']['back'],
                                            transform=transforms.ToTensor() )  
    ValidDataset      = OneshotDataset(args.eval_dir, n_ways=20, idxs_eval=dic_idxs_eval, phase='valid',
                                        transform = transforms.ToTensor() )
    TestDataset       = OneshotDataset(args.eval_dir, n_ways=20, idxs_eval=dic_idxs_eval, phase='test',
                                        transform = transforms.ToTensor() )

    logger.info("***** Train a model from scratch *****")
    model = SiameseNN()
    model.to(args.device)
    
    train(args, BackgroundDataset, ValidDataset, model, hparams)

    """ Load the fine-tuned model for the inference """
    logger.info("***** Loading the fine-tuned model for the test *****")
    model = SiameseNN()
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'model.pt')))
    model.to(args.device)
    model.eval()

    test_acc = oneshot_eval(args, TestDataset, model)
    save_model_and_result(model, args.out_dir, test_acc, 'test')

    return test_acc

def propose_candidates(args, prev_res):

    ## train a surrogate model
    gp = SingleTaskGP(prev_res['hparams'], prev_res['acc'])
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    ## construct an acquisition function and optimize it
    UCB = UpperConfidenceBound(gp, beta=0.1)

    candidates, acq_value = optimize_acqf(UCB, 
                                        bounds=bounds, 
                                        q=args.n_cand, 
                                        num_restarts=5, 
                                        raw_samples=20)

    return candidates


if __name__ == '__main__':
    
    random.seed(10)

    parser = argparse.ArgumentParser()

    parser.add_argument('--back_dir', default='./data/background')
    parser.add_argument('--eval_dir', default='./data/evaluation')
    parser.add_argument('--out_dir', default='./results/tmp')

    parser.add_argument('--train_batch_size', default=40, type = int)
    parser.add_argument('--eval_batch_size', default=20, type = int)
    parser.add_argument('--n_train', default=90000, type = int)
    parser.add_argument('--num_workers', default=1, type = int)
    parser.add_argument('--num_train_epochs', default=200, type = int)
    parser.add_argument('--idx_gpu', default=0, type = int)
    parser.add_argument('--log_step', default=1000, type = int)
    
    parser.add_argument('--bo_iter', default=10, type = int)
    parser.add_argument('--n_cand', default=3, type = int)


    args = parser.parse_args()
    
    args.device = torch.device("cuda", args.idx_gpu)
    
    """assrtion for the arguments"""
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    prev_res = {'hparams' : [],
                'accs'    : []}
    n_iter = 0
    hparams = [INIT_HPARAMS] * args.n_cand
    while n_iter < args.bo_iter:
        accs = []
        for i in range(args.n_cand):
            acc = train_and_test(args, hparams[i])
            accs += [acc]
        

        



        