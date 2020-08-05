import random
import logging
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm, trange

from preprocess import VerificationDataset, split_eval, OneshotDataset
from model import SiameseNN


logger = logging.getLogger(__name__)


def oneshot_eval(args, eval_dataset, model):

    logger.info(f"***** Running {eval_dataset.phase} *****")

    eval_dataloader  = DataLoader(eval_dataset, batch_size = args.eval_batch_size, shuffle=False, num_workers=args.num_workers)    

    total = len(eval_dataset)
    TP = torch.tensor(0, dtype=torch.int64)
    epoch_iterator = tqdm(eval_dataloader, desc="Validation Iteration")
    for step, batch in enumerate(epoch_iterator):
        with torch.no_grad():
            label = batch[2].to(args.device)
            
            scores = torch.tensor([], dtype=torch.float32).to(args.device)
            for i in range(eval_dataset.n_ways):
                inputs = {
                    "img1" : batch[0].to(args.device),
                    "img2" : batch[1][i].to(args.device)
                    }
                outputs = model(**inputs)
                scores = torch.cat((scores, 
                                    outputs.view(1, outputs.size()[0])))
            predicted = torch.argmax(scores, dim=1).type(label.dtype)
            matched = torch.sum(torch.eq(predicted, 
                                         label.view(label.size()[0])))
            TP += matched
    epoch_iterator.close()
    acc = int(TP) / total * 100
    
    return acc
    

def train(args, train_dataset, valid_dataset, model):

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    train_dataloader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle=True, num_workers=args.num_workers)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 3e-2, betas = (.99, .999), weight_decay=.05)

    running_loss = 0.0
    train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):

            model.train()
            optimizer.zero_grad()

            batch = tuple(t.to(args.device) for t in batch)
            labels = batch[2]
            inputs = {
                "img1"  : batch[0],
                "img2"  : batch[1]
                }
            outputs = model(**inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (step % args.log_step) == (args.log_step - 1):
                acc = oneshot_eval(args, valid_dataset, model)
                print(" ")
                print(f"\n \n Validation Accuracy : {acc} / 100.00 \n")
                print('[epoch %d , iteration :%5d] Loss = %.5f \n' %
                            (epoch + 1, step + 1, running_loss / args.log_step))
                running_loss = 0.0


if __name__ == '__main__':
    
    random.seed(10)

    parser = argparse.ArgumentParser()

    parser.add_argument('--back_dir', default='./data/background')
    parser.add_argument('--eval_dir', default='./data/evaluation')

    parser.add_argument('--train_batch_size', default=40, type = int)
    parser.add_argument('--eval_batch_size', default=20, type = int)
    parser.add_argument('--n_train', default=90000, type = int)
    parser.add_argument('--num_workers', default=1, type = int)
    parser.add_argument('--num_train_epochs', default=2, type = int)
    parser.add_argument('--idx_gpu', default=0, type = int)
    parser.add_argument('--log_step', default=1000, type = int)
    

    args = parser.parse_args()
    
    args.device = torch.device("cuda", args.idx_gpu)

    """ Load the datasets, model, and configs """
    logger.info("***** Loading the datasets, model, and configs *****")

    dic_idxs_eval = split_eval(args.back_dir, args.eval_dir)

    BackgroundDataset = VerificationDataset(args.back_dir, n_samples = args.n_train, 
                                            idxs_back = dic_idxs_eval['drawer']['back'],
                                            transform=transforms.ToTensor() )  
    ValidDataset      = OneshotDataset(args.eval_dir, n_ways=20, idxs_eval=dic_idxs_eval, phase='valid',
                                        transform = transforms.ToTensor() )
    TestDataset        = OneshotDataset(args.eval_dir, n_ways=20, idxs_eval=dic_idxs_eval, phase='test',
                                        transform = transforms.ToTensor() )

    model = SiameseNN()
    model.to(args.device)

    train(args, BackgroundDataset, ValidDataset, model)

    oneshot_eval(args, TestDataset, model)
