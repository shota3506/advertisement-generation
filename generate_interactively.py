import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from statistics import mean

from model import AdvertisementGenerator
from dataset import AdvertisementDataset
from search import BeamSearch

parser = argparse.ArgumentParser()
# Data
# parser.add_argument("--data_file", type=str, required=True)
parser.add_argument("--spm_file", type=str, required=True)
# Model
parser.add_argument("--dim_embedding", type=int, default=256)
parser.add_argument("--dim_hidden", type=int, default=512)
parser.add_argument("--checkpoint_path", type=str, required=True)
# Search
parser.add_argument("--beam_size", type=int, default=1)

args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor(model_file=args.spm_file)

    categories=['コスメ', '健康食品', '不動産', '人材', '保険', '金融', 'EC', 'エンタメ', '情報メディア', '理美容・ヘルスケア', '機器', '雑貨', 'アプリ', 'ブライダル', 'ファッション', 'IT・Webサービス', '健康・スポーツ', 'イベント・レジャー', 'その他']
    
    # Build datasets
    model = AdvertisementGenerator(
        num_embeddings=len(sp),
        num_categories=len(categories),
        dim_hidden=args.dim_hidden,
        dim_embedding=args.dim_embedding,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    searcher = BeamSearch(3, beam_size=args.beam_size)

    print('Start!')
    with torch.no_grad():
        # Select Category
        print('-' * 50)
        print('Please select advertisement category\n')
        for i, c in enumerate(categories):
            print("%2d: %s" % (i, c))
        print()
        while 1:
            category = input('Category: ')
            try:
                category = int(category)
                break
            except ValueError:
                print('Please input index of category')

        # Input Keyphrases
        print('-' * 50)
        print('Please input keyphrases\n')
        keyphrases = []
        while 1:
            keyphrase = input("Keyphrae: ")
            if keyphrase == '':
                break
            keyphrases.append(keyphrase)
        
        categories = torch.tensor([category], device=device)
        keyphrases = [[t for w in k.split() for t in sp.encode(w)] for k in keyphrases]
        keyphrases = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(k, device=device) for k in keyphrases], batch_first=True).unsqueeze(0)

        keyphrases_mask = (keyphrases[:, :, 0] == 0)
        keyphrases_padding_mask = (keyphrases == 0)
        memory = model.encode_keyphrase(keyphrases, keyphrases_padding_mask)

        start_predictions = torch.zeros(1, device=device).fill_(2).long()
        start_state = {'memory': memory, 'categories': categories, 'keyphrases_mask': keyphrases_mask}
        predictions, log_probabilities = searcher.search(start_predictions, start_state, model.step)

        print('\nOutputs')
        for pred in predictions[0]:
            pred = pred[(pred != 0) & (pred != 3)].tolist()
            hyp = sp.decode(pred)
            print(hyp)

if __name__ == '__main__':
    main()

