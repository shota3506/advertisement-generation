import os
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

class AdvertisementDataset(Dataset):
    def __init__(self, data_file, spm_file, categories=['コスメ', '健康食品', '不動産', '人材', '保険', '金融', 'EC', 'エンタメ', '情報メディア', '理美容・ヘルスケア', '機器', '雑貨', 'アプリ', 'ブライダル', 'ファッション', 'IT・Webサービス', '健康・スポーツ', 'イベント・レジャー', 'その他']):     
        self.data = data = pd.read_csv(data_file)
        self.sp = spm.SentencePieceProcessor(model_file=spm_file)

        self.categories = categories
        self.category_to_index = {c: i for i, c in enumerate(categories)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data.iloc[idx]

        category = self.category_to_index[d.category]

        text = [BOS_INDEX] + self.sp.encode(d.inside_text) + [EOS_INDEX]

        keyphrase = eval(d.keyphrase)
        keyphrase = [[t for w in k.split() for t in self.sp.encode(w)] for k in keyphrase]

        return {'category': category, 'text': text, 'keyphrase': keyphrase}

    @staticmethod
    def collate_fn(data):
        category = torch.tensor([d['category'] for d in data])

        text = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(d['text']) for d in data], batch_first=True)

        unbatched_keyphrase = [d['keyphrase'] for d in data]
        batch_size = len(data)
        max_keyphrase_length = max(len(k) for k in unbatched_keyphrase)
        max_keyphrase_token_length = max(len(t) for k in unbatched_keyphrase for t in k)

        keyphrase = torch.zeros(batch_size, max_keyphrase_length, max_keyphrase_token_length, dtype=torch.long)
        for i, k in enumerate(unbatched_keyphrase):
            for j, t in enumerate(k):
                keyphrase[i, j, :len(t)] = torch.tensor(t, dtype=torch.long)
        return text, category, keyphrase
