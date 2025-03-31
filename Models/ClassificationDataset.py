
import torch
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, vocab_fname, max_length):
        # Use sentences and labels directly instead of reading from a file
        self.sentences = sentences
        self.labels = labels
        
        self.word_to_id, self.id_to_word = self.make_mapping(vocab_fname)
        
        # Tokenizing the sentences (adding [BOS] and [EOS] tokens)
        self.tokenized = [['[BOS]'] + seq.split(' ') + ['[EOS]'] for seq in self.sentences]

        self.encoded = [self.encode(seq) for seq in self.tokenized]

        self.X = [torch.tensor(seq) for seq in self.encoded]
        self.y = torch.tensor(self.labels)  #  Each

        self.max_length = max_length



    def make_mapping(self, vocab_fname):
        with open(vocab_fname, 'r') as f:
            vocab = f.read().split()
            vocab = set([word.strip() for word in vocab])

        special_tokens = {
            '[PAD]': 0,
            '[BOS]': len(vocab)+1,
            '[EOS]': len(vocab)+2
        }
                

        word_to_id = {}
        id_to_word = {}
        
        
        id = 1
        
        for i in vocab:
            word_to_id[i] = id
            id_to_word[id] = i
            id += 1
        
        
        id_to_word = id_to_word | special_tokens
        word_to_id =  word_to_id | special_tokens
        

        ## Build word_to_id and id_to_word
        ## Ids for words should start with 1 because [PAD] is 0. 

        return word_to_id, id_to_word

    def encode(self, seq):
        """Encodes sequence of words with IDs
        """
        encoded_seq = []
        for word in seq:
            if word in self.word_to_id:  # Only add known words
                encoded_seq.append(self.word_to_id[word])
        return encoded_seq

    def decode(self, seq):
        """Decodes sequence of IDs to words
        """
        decoded_seq = []
        for idx in seq:
            if idx in self.id_to_word:  # Only add known IDs
                decoded_seq.append(self.id_to_word[idx])
        return decoded_seq


    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.y[idx]

        ## Left pad
        padded_x = torch.full((1,self.max_length), self.word_to_id['[PAD]'], dtype=torch.int).flatten()
        padded_x[-x.size(0):] = x

        return padded_x, y


    def __len__(self):
        return len(self.X)