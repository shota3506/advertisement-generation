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
# Model
parser.add_argument("--dim_embedding", type=int, default=256)
parser.add_argument("--dim_hidden", type=int, default=512)
parser.add_argument("--checkpoint_path", type=str, required=True)
# Search
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--beam_size", type=int, default=1)

args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets
    dataset = AdvertisementDataset(args.data_file, args.spm_file)
    indices = list(range(len(dataset)))
    sampler = SequentialSampler(indices[int(0.9 * len(dataset)):])
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
        dim_hidden=args.dim_hidden,
        dim_embedding=args.dim_embedding,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    searcher = BeamSearch(3, beam_size=args.beam_size)

    with torch.no_grad():
        for txts, categories, keyphrases in loader:
            categories, keyphrases = categories.to(device), keyphrases.to(device)
            keyphrases_mask = (keyphrases[:, :, 0] == 0)
            keyphrases_padding_mask = (keyphrases == 0)

            bsz = len(categories)
            memory = model.encode_keyphrase(keyphrases, keyphrases_padding_mask)

            start_predictions = torch.zeros(bsz, device=device).fill_(2).long()
            start_state = {'memory': memory, 'categories': categories, 'keyphrases_mask': keyphrases_mask}

            predictions, log_probabilities = searcher.search(start_predictions, start_state, model.step)
            for i in range(len(predictions)):
                print("Category:", dataset.categories[categories[i].item()])
                
                kps = [kp.tolist() for kp in keyphrases[i] if kp[0] != 0]
                print("Keyphrases:", ', '.join([dataset.sp.decode(kp) for kp in kps]))

                preds = predictions[i]
                for pred in preds:
                    pred = pred[(pred != 0) & (pred != 3)].tolist()
                    hyp = dataset.sp.decode(pred)
                    print(hyp)
                print()


if __name__ == '__main__':
    main()

