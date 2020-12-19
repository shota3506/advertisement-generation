import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

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
# parser.add_argument("--use_pretrained", type=bool, default=True)
# parser.add_argument("--checkpoint_path", type=str, required=True)
# # Optim
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=0.001)

args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
g        collate_fn=dataset.collate_fn,
    )
    
    # Build datasets
    model = AdvertisementGenerator(
        num_embeddings=len(dataset.sp),
        num_categories=len(dataset.categories),
        dim_model=args.dim_model,
    ).to(device)
    
    criterion = LabelSmoothedLmCrossEntropyLoss(0, label_smoothing=0.1, reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09)

    for i_epoch in range(args.num_epochs):
        pbar = tqdm(train_loader)
        pbar.set_description("Epoch %d" % (i_epoch+1))
        model.train()
        for text, category, keyphrase in pbar:
            text, category, keyphrase = text.to(device), category.to(device), keyphrase.to(device)
            keyphrase_mask = (keyphrase[:, :, 0] == 0)
            keyphrase_padding_mask = (keyphrase == 0)

            output, _ = model(text, category, keyphrase, keyphrase_mask, keyphrase_padding_mask)
            loss = criterion(output[:, :-1, :], text[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())

        model.eval()
        with torch.no_grad():
            pbar = valid_loader
            for text, category, keyphrase in pbar:
                text, category, keyphrase = text.to(device), category.to(device), keyphrase.to(device)
                keyphrase_mask = (keyphrase[:, :, 0] == 0)
                keyphrase_padding_mask = (keyphrase == 0)

                output, _ = model(text, category, keyphrase, keyphrase_mask, keyphrase_padding_mask)
                loss = criterion(output[:, :-1, :], text[:, 1:])

        torch.save(model.state_dict(), args.checkpoint_path)

if __name__ == '__main__':
    main()

