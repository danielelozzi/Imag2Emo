import math
import torch
import torch.nn as nn
import torch.nn.functional as F

##############################
# Helper: MLP block
##############################
def make_mlp(input_dim, hidden_units, dropout_rate):
    """
    Returns a sequential MLP module.
    
    hidden_units: an iterable of hidden layer sizes.
    Note: In the Keras code, each Dense uses ELU activation.
    """
    layers_list = []
    in_dim = input_dim
    # Se hidden_units è un intero, lo racchiudiamo in una lista:
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    for units in hidden_units:
        layers_list.append(nn.Linear(in_dim, units))
        layers_list.append(nn.ELU())
        layers_list.append(nn.Dropout(dropout_rate))
        in_dim = units
    return nn.Sequential(*layers_list), in_dim  # restituisce anche la dimensione finale

##############################
# PatchEncoder module
##############################
class PatchEncoder(nn.Module):
    def __init__(self, num_patches, in_dim, projection_dim):
        """
        Proietta i patch (di dimensione in_dim) in uno spazio di dimensione projection_dim,
        aggiungendo un embedding posizionale appreso.
        """
        super(PatchEncoder, self).__init__()
        self.projection = nn.Linear(in_dim, projection_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(num_patches, projection_dim)
        )

    def forward(self, x):
        # x: (batch, num_patches, in_dim)
        x = self.projection(x)  # (batch, num_patches, projection_dim)
        # Sommiamo l'embedding posizionale (broadcasting sul batch)
        x = x + self.position_embedding.unsqueeze(0)
        return x

##############################
# Transformer Block module
##############################
class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads, transformer_units, dropout_attention=0.5, dropout_mlp=0.7):
        """
        transformer_units: lista (o int) delle dimensioni nascoste per l'MLP interno al trasformatore.
        """
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(projection_dim)
        self.attn = nn.MultiheadAttention(embed_dim=projection_dim,
                                          num_heads=num_heads,
                                          dropout=dropout_attention,
                                          batch_first=False)  # attende (seq, batch, embed_dim)
        self.norm2 = nn.LayerNorm(projection_dim)
        # Creiamo l'MLP per il blocco transformer.
        self.mlp, _ = make_mlp(projection_dim, transformer_units, dropout_mlp)

    def forward(self, x):
        # x: (batch, num_patches, projection_dim)
        # Per l'attenzione multi-head servono (seq, batch, embed_dim)
        x_in = x
        x_norm = self.norm1(x)
        x_attn, attn_weights = self.attn(x_norm.transpose(0,1), 
                                         x_norm.transpose(0,1),
                                         x_norm.transpose(0,1),
                                         need_weights=True)
        # x_attn ha forma (seq, batch, projection_dim)
        x_attn = x_attn.transpose(0,1)  # torna a (batch, seq, projection_dim)
        x = x_in + x_attn  # skip connection
        
        x_in2 = x
        x_norm2 = self.norm2(x)
        x_mlp = self.mlp(x_norm2)
        x = x_in2 + x_mlp  # seconda skip connection
        return x, attn_weights

##############################
# Il modello ConTraNet in PyTorch
##############################
class ConTraNet(nn.Module):
    def __init__(self,
                 nb_classes,
                 Chans,           # numero di canali EEG
                 Samples,         # lunghezza della sequenza
                 dropoutRate,     # dropout rate per il blocco CNN (es. 0.5)
                 regRate,         # regolarizzazione (non usata in questa versione PyTorch)
                 kernLength,      # lunghezza del kernel per la conv temporale (es. sampling_frequency/2)
                 poolLength,      # dimensione del pooling nel tempo (es. 8)
                 numFilters,      # numero di filtri convoluzionali (F1)
                 dropoutType,     # stringa: 'SpatialDropout2D' oppure 'Dropout'
                 projection_dim,  # dimensione di proiezione per il patch encoder (es. 32)
                 transformer_layers,  # numero di blocchi transformer (es. 1)
                 num_heads,           # numero di teste nell'attenzione multi-head (es. 8)
                 transformer_units,   # lista (o int) delle unità per l'MLP del trasformatore (es. [projection_dim*2, projection_dim])
                 mlp_head_units,      # lista (o int) delle unità per l'MLP finale (es. 112)
                 training=True):
        """
        Il modello è composto da:
          - Un blocco CNN che applica una convoluzione temporale sull'input EEG
          - Un modulo di patch extraction e patch encoding che suddivide la mappa di feature in patch,
            le proietta e aggiunge embedding posizionali.
          - Una serie di blocchi Transformer.
          - Un MLP finale per la classificazione.
          
        L'input atteso dal modello è di forma (batch, channels, samples),
        dove 'channels' corrisponde a Chans.
        """
        super(ConTraNet, self).__init__()
        self.training_flag = training  # se False, si possono restituire informazioni extra (es. pesi di attenzione)

        # 1. Blocco CNN
        # Per utilizzare Conv2d, aggiungiamo una dimensione dummy in ingresso.
        # Kernel size: (1, kernLength) per applicare la convoluzione solo sul tempo.
        pad_width = kernLength // 2
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=numFilters,
                              kernel_size=(1, kernLength),
                              padding=(0, pad_width),
                              bias=False)
        self.bn = nn.BatchNorm2d(numFilters)
        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, poolLength))
        # Selezione del dropout: se 'SpatialDropout2D' si usa nn.Dropout2d, altrimenti nn.Dropout.
        if dropoutType == 'SpatialDropout2D':
            self.dropout_cnn = nn.Dropout2d(dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropout_cnn = nn.Dropout(dropoutRate)
        else:
            raise ValueError("dropoutType deve essere 'SpatialDropout2D' o 'Dropout'")

        # 2. Blocco Transformer
        # Dopo il blocco CNN, la mappa delle feature ha forma:
        #    (batch, numFilters, Chans, T)   dove T è approssimativamente Samples/poolLength.
        # Effettuiamo l'extraction dei patch sui dati.
        # Ipotizziamo di estrarre patch che coprono l'intera dimensione dei canali (Chans)
        # con patch_width = 2 e stride (Chans, 1).
        self.Chans = Chans  # serve per l'estrazione dei patch
        self.numFilters = numFilters
        # Utilizziamo nn.Unfold con kernel_size=(Chans,2) e stride=(Chans,1)
        self.patch_size = (Chans, 2)
        self.patch_stride = (Chans, 1)
        # Calcoliamo T, la dimensione temporale dopo pooling:
        T = Samples // poolLength  # approssimato
        # Numero di patch: con kernel (Chans,2) e stride (Chans,1) sulla dimensione temporale T:
        num_patches = (T - 2) // 1 + 1 if T >= 2 else 0
        self.num_patches = num_patches
        # Ogni patch ha dimensione: numFilters * (Chans * 2)
        self.patch_dim = numFilters * (Chans * 2)

        # Patch encoder: proietta ogni patch in projection_dim e aggiunge embedding posizionale.
        self.patch_encoder = PatchEncoder(num_patches, self.patch_dim, projection_dim)

        # Blocchi Transformer:
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, transformer_units, dropout_attention=0.5, dropout_mlp=0.7)
            for _ in range(transformer_layers)
        ])
        # Layer normalization finale dopo i blocchi Transformer:
        self.transf_norm = nn.LayerNorm(projection_dim)

        # 3. MLP finale
        # Dopo i blocchi Transformer, appiattiamo i dati: dimensione totale = num_patches * projection_dim.
        mlp_input_dim = num_patches * projection_dim
        self.mlp_head, mlp_output_dim = make_mlp(mlp_input_dim, mlp_head_units, dropout_rate=0.7)
        
        # Cambiamo la funzione di attivazione finale per evitare Softmax
        self.classifier = nn.Linear(mlp_output_dim, nb_classes)

        # Applichiamo ELU prima dello softmax (come nel modello Keras originale).
        self.final_activation = nn.ELU()

    def forward(self, x):
        """
        x è atteso di forma (batch, channels, samples)
        """
        # Se l'input ha 3 dimensioni, aggiungiamo una dimensione "dummy" per i canali 2D
        if x.dim() == 3:
            # Da (batch, channels, samples) a (batch, 1, channels, samples)
            x = x.unsqueeze(1)
            
        # Blocco CNN
        x = self.conv(x)             # -> (batch, numFilters, Chans, Samples) (con padding lungo il tempo)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)             # -> (batch, numFilters, Chans, T)
        x = self.dropout_cnn(x)

        # Estrazione dei patch:
        # x ha forma (batch, numFilters, Chans, T).
        # Utilizziamo nn.Unfold per estrarre patch con kernel (Chans,2) e stride (Chans,1).
        # L'output di nn.Unfold ha forma (batch, patch_dim, num_patches)
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_stride)
        patches = unfold(x)  # (batch, patch_dim, num_patches)
        patches = patches.transpose(1, 2)  # ora (batch, num_patches, patch_dim)

        # Patch encoding: proiezione e aggiunta dell'embedding posizionale
        encoded = self.patch_encoder(patches)  # (batch, num_patches, projection_dim)

        # Blocchi Transformer:
        for block in self.transformer_blocks:
            encoded, attn_weights = block(encoded)
        encoded = self.transf_norm(encoded)
        # Appiattiamo l'output del trasformatore:
        batch_size = encoded.size(0)
        encoded = encoded.reshape(batch_size, -1)  # (batch, num_patches * projection_dim)
        encoded = F.dropout(encoded, p=0.5, training=self.training)

        # MLP finale:
        features = self.mlp_head(encoded)
        # Rimuoviamo la softmax (se usi CrossEntropyLoss)
        logits = self.classifier(features)  # (batch_size, nb_classes)
        
        

        #logits = self.classifier(features)
        #logits = self.final_activation(logits)
        # Softmax per ottenere le probabilità di classe
        #output = F.softmax(logits, dim=-1)
        #if not self.training_flag:
            # Se non siamo in training, possiamo restituire anche i pesi di attenzione.
        #    return output, attn_weights
        #return output

        if not self.training_flag:
            # Se non siamo in training, possiamo restituire anche i pesi di attenzione.
            return logits, attn_weights

        # Se stai usando CrossEntropyLoss, NON serve Softmax
        return logits  # Output senza softmax, poiché la loss lo gestisce internamente


