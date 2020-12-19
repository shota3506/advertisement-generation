import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from statistics import mean

from model import AdvertisementGenerator
from dataset import AdvertisementDataset
from criterion import LmCrossEntropyLoss, LabelSmoothedLmCrossEntropyLoss

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_file", type=str, required=True)
parser.add_argument("--spm_file", type=str, required=True)
parser.add_argument("--split_rate", type=float, default=0.2)
# # Model
parser.add_argument("--dim_model", type=int, default=256)
parser.add_argument("--checkpoint_path", type=str, required=True)
# # Optim
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--log_file", type=str, required=True)

args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(filename=args.log_file, mode='w')
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
    handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    # Build datasets
    dataset = AdvertisementDataset(args.data_file, args.spm_file)

    indices = list(range(len(dataset)))
    split = int((1 - args.split_rate) *  len(dataset))
    train_sampler = SubsetRandomSampler(indices[:split])
    valid_sampler = SubsetRandomSampler(indices[split:])

    train_loader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    valid_loader = DataLoader(
        dataset,
        sampler=valid_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )
    
    # Build datasets
    model = AdvertisementGenerator(
        num_embeddings=len(dataset.sp),
        num_categories=len(dataset.categories),
        dim_model=args.dim_model,
    ).to(device)
    
    criterion = LabelSmoothedLmCrossEntropyLoss(0, label_smoothing=0.1, reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09)

    logger.info('Start training')
    for i_epoch in range(args.num_epochs):
        bookkeeper = {'Train loss': [], 'Valid loss': []}
        pbar = tqdm(train_loader)
        pbar.set_description("Epoch %d" % (i_epoch+1))
        model.train()
        for txts, categories, keyphrases in pbar:
            txts, categories, keyphrases = txts.to(device), categories.to(device), keyphrases.to(device)
            keyphrases_mask = (keyphrases[:, :, 0] == 0)
            keyphrases_padding_mask = (keyphrases == 0)

            output, _ = model(txts, categories, keyphrases, keyphrases_mask, keyphrases_padding_mask)
            loss = criterion(output[:, :-1, :], txts[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bookkeeper['Train loss'].append(loss.item())

        model.eval()
        with torch.no_grad():
            pbar = valid_loader
            for txts, categories, keyphrases in pbar:
                txts, categories, keyphrases = txts.to(device), categories.to(device), keyphrases.to(device)
                keyphrases_mask = (keyphrases[:, :, 0] == 0)
                keyphrases_padding_mask = (keyphrases == 0)

                output, _ = model(txts, categories, keyphrases, keyphrases_mask, keyphrases_padding_mask)
                loss = criterion(output[:, :-1, :], txts[:, 1:])

                bookkeeper['Valid loss'].append(loss.item())

        logger.info('[Epoch %d] Train loss %.4f, Valid loss %.4f' % (i_epoch, mean(bookkeeper['Train loss']), mean(bookkeeper['Valid loss'])))
        torch.save(model.state_dict(), args.checkpoint_path)

if __name__ == '__main__':
    main()

