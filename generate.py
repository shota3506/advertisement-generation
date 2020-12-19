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
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from statistics import mean

from model import AdvertisementGenerator
from dataset import AdvertisementDataset
from search import BeamSearch

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_file", type=str, required=True)
parser.add_argument("--spm_file", type=str, required=True)
# # Model
parser.add_argument("--dim_model", type=int, default=256)
parser.add_argument("--checkpoint_path", type=str, required=True)
# # Optim
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=1)

args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets
    dataset = AdvertisementDataset(args.data_file, args.spm_file)
    indices = list(range(len(dataset)))
    sampler = SequentialSampler(indices[-100:])
    loader = DataLoader(
        dataset,
        sampler=sampler,
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
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    searcher = BeamSearch(3, beam_size=1)

    print('start!')

    with torch.no_grad():
        for _, categories, keyphrases in tqdm(loader):
            categories, keyphrases = categories.to(device), keyphrases.to(device)
            keyphrases_mask = (keyphrases[:, :, 0] == 0)
            keyphrases_padding_mask = (keyphrases == 0)

            bsz = len(categories)
            memory = model.encode_keyphrase(keyphrases, keyphrases_padding_mask)

            start_predictions = torch.zeros(bsz, device=device).fill_(2).long()
            start_state = {'memory': memory, 'categories': categories, 'keyphrases_mask': keyphrases_mask}

            predictions, log_probabilities = searcher.search(start_predictions, start_state, model.step)
            for pred in predictions:
                pred = pred[(pred != 0) & (pred != 3)].tolist()
                hyp = dataset.sp.decode(pred)
                print(hyp)


if __name__ == '__main__':
    main()

