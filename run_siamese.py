import os
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
from datetime import datetime

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
    
    print(f"\n \n {eval_dataset.phase}" + " Accuracy : {:.5} / 100.00 \n".format(acc))
    
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
    best_val = {'epoch' : 0, 'acc' : 0.0}
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
                print(" - ")
                print('\n [epoch %d , iteration :%5d] Loss = %.5f' %
                            (epoch + 1, step + 1, running_loss / args.log_step))
                running_loss = 0.0

        val_acc = oneshot_eval(args, valid_dataset, model)
        # Update best epoch
        if val_acc > best_val['acc']:
            logger.info('Best epoch is updated')
            best_val['epoch'] = epoch
            best_val['acc'] = val_acc
            output_dir = os.path.join(args.out_dir, 'checkpoint-{}'.format(epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
            with open(os.path.join(output_dir, 'val_acc.txt'), 'w') as f:
                f.write(f'Validation Accuracy : {val_acc}')
        # Termination
        elif (epoch - best_val['epoch'] > 20) or (epoch >= args.num_train_epochs):
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))
            with open(os.path.join(args.out_dir, 'val_acc.txt'), 'w') as f:
                f.write(f'Best Epoch : {epoch} \n Validation Accuracy : {val_acc}')
            break

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
    parser.add_argument('--log_step', default=1000, type = int)11
    

    args = parser.parse_args()
    
    args.device = torch.device("cuda", args.idx_gpu)
    
    """assrtion for the arguments"""
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    start = datetime.now()

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

    model = SiameseNN()
    model.to(args.device)

    train(args, BackgroundDataset, ValidDataset, model)

    test_acc = oneshot_eval(args, TestDataset, model)

    print('Consumed Time Total %s' %(str(datetime.now()-start)))