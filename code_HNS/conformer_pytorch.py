import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------
# 1. Patch Embedding
# ----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, n_chan=22, patch_size=25, pool_size=75, pool_stride=15, num_shallow_filters=40, dropout_rate=0.5):
        """
        Crea i patch embedding a partire dai segnali EEG.
        
        Args:
            emb_size (int): Dimensione dell'embedding finale.
            n_chan (int): Numero di canali EEG.
            patch_size (int): Dimensione del kernel per la prima convoluzione (lunghezza lungo il tempo).
            pool_size (int): Dimensione della finestra di pooling.
            pool_stride (int): Stride del pooling.
            num_shallow_filters (int): Numero di filtri nella parte convoluzionale.
            dropout_rate (float): Tasso di dropout.
        """
        super(PatchEmbedding, self).__init__()
        # Convoluzione per catturare feature locali lungo il tempo
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_shallow_filters, kernel_size=(1, patch_size), stride=(1, 1))
        # Convoluzione che unisce le informazioni da tutti i canali
        self.conv2 = nn.Conv2d(in_channels=num_shallow_filters, out_channels=num_shallow_filters, kernel_size=(n_chan, 1), stride=(1, 1))
        self.batchnorm = nn.BatchNorm2d(num_shallow_filters)
        self.elu = nn.ELU()
        # Pooling per "slicing" lungo la dimensione temporale (simile alla creazione dei patch in ViT)
        self.avgpool = nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout_rate)
        # Proiezione per ottenere l'embedding finale
        self.proj_conv = nn.Conv2d(in_channels=num_shallow_filters, out_channels=emb_size, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor di forma (batch, n_chan, tempo)
        Returns:
            Tensor di forma (batch, n_patches, emb_size)
        """
        # Aggiungiamo una dimensione per il canale (necessaria per Conv2d)
        x = x.unsqueeze(1)  # (batch, 1, n_chan, tempo)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.proj_conv(x)
        # Rimodelliamo da (batch, emb_size, h, w) a (batch, h*w, emb_size)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x

# ----------------------------
# 2. Multi-Head Attention
# ----------------------------
'''class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Proiezione lineare per ottenere queries, keys e values
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        # Riorganizzazione per separare le teste: (batch, num_heads, n_tokens, emb_size//num_heads)
        queries = rearrange(queries, "b n (h d) -> b h n d", h=self.num_heads)
        keys    = rearrange(keys,    "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(values,  "b n (h d) -> b h n d", h=self.num_heads)
        # Calcolo dei punteggi di attenzione
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)
        scaling = math.sqrt(self.emb_size)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        # Applicazione dell'attenzione ai values
        out = torch.einsum('bhqk, bhkd -> bhqd', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out'''


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        # Variabile per salvare la mappa di attenzione dell'ultima forward
        self.last_att = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Proiezioni lineari e riorganizzazione per le teste multiple
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys    = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        
        # Calcolo dei punteggi (energy)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)
        
        scaling = math.sqrt(self.emb_size)
        att = F.softmax(energy / scaling, dim=-1)
        # Salviamo la mappa di attenzione (detach la portiamo su CPU per visualizzazione)
        self.last_att = att.detach().cpu()
        
        att = self.att_drop(att)
        out = torch.einsum('bhqk, bhkd -> bhqd', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out



# ----------------------------
# 3. Feed-Forward Block
# ----------------------------
class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion, drop_p):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(emb_size, expansion * emb_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(drop_p)
        self.fc2 = nn.Linear(expansion * emb_size, emb_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ----------------------------
# 4. Transformer Encoder Block
# ----------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super(TransformerEncoderBlock, self).__init__()
        # Blocco di multi-head attention con connessione residua
        self.norm1 = nn.LayerNorm(emb_size)
        self.attention = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.dropout1 = nn.Dropout(drop_p)
        # Blocco feed-forward con connessione residua
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.dropout2 = nn.Dropout(drop_p)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Residual connection per l'attenzione
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        # Residual connection per il feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)
        return x

# ----------------------------
# 5. Transformer Encoder
# ----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        """
        Stack di blocchi Transformer.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, drop_p, forward_expansion, forward_drop_p)
            for _ in range(depth)
        ])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x

# ----------------------------
# 6. Classification Head
# ----------------------------
class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes, n_hidden):
        """
        Head per la classificazione basata su un MLP.
        
        Args:
            emb_size (int): Dimensione dell'embedding.
            n_classes (int): Numero di classi.
            n_hidden (int): Dimensione della feature map appiattita in ingresso.
        """
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(n_hidden, 256)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 32)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Appiattiamo la feature map: (batch, n_patches, emb_size) -> (batch, n_hidden)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.elu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.elu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ----------------------------
# 7. Conformer (il modello completo)
# ----------------------------
class Conformer(nn.Module):
    def __init__(self, n_chan, n_classes, emb_size=40, depth=6, n_hidden=2440, n_heads=10,
                 patch_size=25, pool_size=75, pool_stride=15, num_shallow_filters=40,
                 drop_p=0.5, forward_expansion=4, forward_drop_p=0.5, **kwargs):
        """
        Modello Conformer per EEG decoding.
        
        Args:
            n_chan (int): Numero di canali EEG.
            n_classes (int): Numero di classi.
            emb_size (int): Dimensione dell'embedding.
            depth (int): Profondità (numero di blocchi Transformer).
            n_hidden (int): Dimensione della feature map appiattita per la classificazione.
            n_heads (int): Numero di teste nell'attenzione multi-head.
            patch_size (int): Kernel size per la prima conv in PatchEmbedding.
            pool_size (int): Finestra di pooling in PatchEmbedding.
            pool_stride (int): Stride del pooling in PatchEmbedding.
            num_shallow_filters (int): Numero di filtri nella parte convoluzionale.
            drop_p (float): Dropout rate per l'attenzione.
            forward_expansion (int): Fattore di espansione nel blocco feed-forward.
            forward_drop_p (float): Dropout rate nel blocco feed-forward.
            **kwargs: Argomenti aggiuntivi.
        """
        super(Conformer, self).__init__()
        self.patch_embedding = PatchEmbedding(
            emb_size=emb_size,
            n_chan=n_chan,
            patch_size=patch_size,
            pool_size=pool_size,
            pool_stride=pool_stride,
            num_shallow_filters=num_shallow_filters
        )
        self.transformer_encoder = TransformerEncoder(
            depth=depth,
            emb_size=emb_size,
            num_heads=n_heads,
            drop_p=drop_p,
            forward_expansion=forward_expansion,
            forward_drop_p=forward_drop_p
        )
        self.classification_head = ClassificationHead(
            emb_size=emb_size,
            n_classes=n_classes,
            n_hidden=n_hidden
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x, mask)
        x = self.classification_head(x)
        return x
'''
class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        self.root = '/Data/strict_TE/'

        self.log_write = open("./results/log_subject%d.txt" % self.nSub, "w")


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        # summary(self.model, (1, 22, 1000))


    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):

        # train data
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]


        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel
'''

'''    def train(self):

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))


                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # out_epoch = time.time()


            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)


                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred


        torch.save(self.model.module.state_dict(), 'model.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def main():
    best = 0
    aver = 0
    result_write = open("./results/sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()


        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


        print('Subject %d' % (i+1))
        exp = ExP(i + 1)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 9
    aver = aver / 9

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close() '''

