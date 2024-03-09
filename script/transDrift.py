import arch as arch
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(arch.log_path, name="my_model")

import os
import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    get_ipython().system('pip install --quiet pytorch-lightning>=1.4')
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.optim as optim
from util.transformer import *
from torch.autograd import Variable
# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE: ", device)
# Read common words present in both timestamp and return list of dictionary.
# Dict1 contains common words present in D_t and D_{t+1}. 
# Dict2 contains common words present in D_t and D_{t+1}_small.
def get_words(file):
    Dict1 = {}
    Dict2 = {}
    list_Dict = []
    
    f = open(arch.COMMON_WORDS, 'r')
    lines = f.readlines()
    index = 0
    for line in lines:
        Dict1[line[:-1]] = index
        index += 1
    f.close()
    
    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        Dict2[line[:-1]] = index
        index += 1
    f.close()
    
    list_Dict.append(Dict1)
    list_Dict.append(Dict2)
    return list_Dict

# words is a list of dictionary containing common words
# present in (D_t, D_{t+1}) and (D_t, D{t+1}_small).
words = get_words(arch.COMMON_WORDS_D2_SMALL)

# rev_words is index to word mapping of common words.
# e.g. words[0]: {sky, 0}, rev_words: {0, sky} 
rev_words = {}
common_dict = words[0]
for key in common_dict:
    rev_words[common_dict[key]] = key
length_common = len(words[0])
length_small = len(words[1])

class DataSet(torch.utils.data.Dataset):
    def __init__(self, words, num_ex, val):
        self.words_dict = words
        self.num_ex = num_ex
        self.adder = val
    def __len__(self):
        return self.num_ex

    def __getitem__(self, index):
        # Fetching word embeddings of D_t, D_{t+1}_small and D_{t+1}.
        f_D1_full = open(arch.DATASET + str(index+1+self.adder) + arch.D1, 'r')
        f_D2_small = open(arch.DATASET + str(index+1+self.adder) + arch.D2_SMALL, 'r')
        f_D2_full = open(arch.DATASET + str(index+1+self.adder) + arch.D2, 'r')

        # inp is a 2d-array where each row represent 50-dimensional word embedding
        # of a word present in common_words.
        # label contains word embeddings of common_words present at time stamp {t+1}.
        inp = [[0.0 for i in range(50)] for i in range(length_common+length_small)]
        label = [[0.0 for i in range(50)] for i in range(length_common+length_small)]
        
        # common words present in D_t and D_{t+1}.
        common_words = self.words_dict[0]
        # common words present in D_t and D_{t+1}_small.
        small_words = self.words_dict[1]

        emb_D1_full = f_D1_full.readlines()[1:]
        for line in emb_D1_full:
            l = line.split()
            if l[0] in common_words:
                vect = np.array(l[1:]).astype(np.float32)
                inp[self.words_dict[0][l[0]]] = vect

        emb_D2_small = f_D2_small.readlines()[1:]
        for line in emb_D2_small:
            l = line.split()
            if l[0] in small_words:
                vect = np.array(l[1:]).astype(np.float32)
                inp[self.words_dict[1][l[0]]] = vect 

        emb_D2_full =  f_D2_full.readlines()[1:]
        for line in emb_D2_full:
            l = line.split()
            vect = np.array(l[1:]).astype(np.float32)
            if l[0] in common_words:
                label[self.words_dict[0][l[0]]] = vect 
            if (l[0] in small_words):
                label[self.words_dict[1][l[0]]] = vect 

        return torch.tensor(inp, device=device), torch.tensor(label, device=device)

# 800 instances for training, 100 for validation, 100 for testing.
batch_size = arch.BATCH_SIZE
train_loader = data.DataLoader(DataSet(words, 800, 0), batch_size=batch_size, shuffle=arch.SHUFFLE_TRAIN, drop_last=True)
val_loader = data.DataLoader(DataSet(words, 100, 800), batch_size=batch_size, shuffle=arch.SHUFFLE_VAL, drop_last=True)
test_loader = data.DataLoader(DataSet(words, 100, 900), batch_size=batch_size, shuffle=arch.SHUFFLE_TEST, drop_last=True)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerPredictor(pl.LightningModule):

    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0):
        
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(max_len = length_common+length_small, d_model=self.hparams.model_dim)

        # Transformer
        self.transformer = TransformerEncoder(num_blocks=self.hparams.num_layers,
                                              d_model=self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )

    def forward(self, x, mask=None, add_positional_encoding=False):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=False):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
baseline_total = 0
baseline_count = 0
test_count = 0
epoch_count = 0

class EmbeddingPredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        global baseline_total, baseline_count, test_count, epoch_count
        # Fetch data and transform categories to one-hot vectors
        inp_data, labels = batch
      
        # Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, add_positional_encoding=arch.POSITIONAL_ENCODING)
        loss_func = nn.CosineEmbeddingLoss().cuda()
        pred_m = preds[:,0:length_common,:]
        labels_m = labels[:,0:length_common,:]

        # Baseline
        check_data = inp_data[:,0:length_common,:]
        baseline_ = cos(check_data.contiguous().view(-1, check_data.size(-1)), labels_m.contiguous().view(-1, labels_m.size(-1))).mean()
        baseline_total += baseline_
        baseline_count += 1
        
        loss = loss_func(preds.view(-1, preds.size(-1)), labels.view(-1, labels.size(-1)), Variable(torch.Tensor(preds.size(0)*preds.size(1)).cuda().fill_(1.0)))
        # writer.add_scalar('TRAIN/loss', loss, global_step)
        acc = cos(pred_m.contiguous().view(-1, pred_m.size(-1)), labels_m.contiguous().view(-1, labels_m.size(-1))).mean()
        
        # Logging
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        if (mode == "test" or mode == "val"):
            epoch_count += 1
        if (mode == "test"):
            if (arch.SAVE_PRED):
                try:
                    os.mkdir(arch.PRED_FOLDER)
                except:
                    pass
                for i in range(batch_size):
                    f = open("temp.txt", 'w')
                    np.savetxt(f, pred_m[i].cpu())
                    f.close()
                    f = open("temp.txt", 'r')
                    lines = f.readlines()
                    f1 = open(arch.PRED_FOLDER + "/pred_"+str(batch_size*test_count +i+1)+".txt", 'w')
                    ind_word = 0
                    for line in lines:
                        f1.write(rev_words[ind_word] + " ")
                        f1.write(line)
                        ind_word += 1
                    f1.close()
                    f.close()
                    
                    #Saving labels
                    f2 = open("temp.txt", 'w')
                    np.savetxt(f2, labels_m[i].cpu())
                    f2.close()
                    f2 = open("temp.txt", 'r')
                    lines = f2.readlines()
                    f3 = open(arch.PRED_FOLDER + "/label_"+str(batch_size*test_count +i+1)+".txt", 'w')
                    ind_word = 0
                    for line in lines:
                        f3.write(rev_words[ind_word] + " ")
                        f3.write(line)
                        ind_word += 1
                    f3.close()
                    f2.close()
                test_count += 1
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")

def train_reverse(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = "output"
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus= 1 if str(device).startswith("cuda") else 0,
                         max_epochs = arch.EPS,
                         gradient_clip_val = arch.GRADIENT_CLIP_VAL,
                         progress_bar_refresh_rate=1, logger = logger)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    model = EmbeddingPredictor(max_iters=trainer.max_epochs*len(train_loader), **kwargs)
    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"]}

    model = model.to(device)
    return model, result

reverse_model, reverse_result = train_reverse(input_dim = arch.INPUT,
                                              model_dim = arch.MODEL,
                                              num_heads = arch.HEADS,
                                              num_classes = arch.CLASSES,
                                              num_layers = arch.LAYERS,
                                              dropout = arch.DROPOUT,
                                              lr = arch.LR,
                                              warmup = arch.WARMUP)

print("Baseline_avg: ", baseline_total/baseline_count)
print(f"Test accuracy: {(100.0 * reverse_result['test_acc']):4.2f}%")

f_acc = open(arch.log_path + arch.D2_SMALL[:-4] + "_test_accuracy.txt", "w")
f_acc.write("Baseline_avg: " +  str(baseline_total/baseline_count) + "\n")
f_acc.write(f"Test accuracy: {(100.0 * reverse_result['test_acc']):4.2f}%")
f_acc.close()
