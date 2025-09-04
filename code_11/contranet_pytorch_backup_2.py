# con_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################
# Funzione MLP (multi-layer perceptron)
############################################
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate, reg_rate=None):
        """
        Se hidden_units è un intero lo converte in lista (altrimenti si assume sia già una lista).
        reg_rate è passato per completezza (per il max norm) ma non viene usato internamente.
        """
        super(MLP, self).__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        layers = []
        in_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

############################################
# Modulo per l'estrazione delle patch
############################################
class Patches(nn.Module):
    def __init__(self, patch_height, patch_width):
        """
        Estrae patch usando nn.Unfold.
        L’idea è quella di estrarre patch di dimensione (patch_height, patch_width)
        dallo spazio (height, width) della mappa di attivazione.
        
        Nel nostro caso, dopo il blocco CNN l'output ha forma:
          (batch, numFilters, Chans, new_samples)
        e vogliamo estrarre patch che coprano l'intera dimensione dei canali (Chans)
        e una finestra temporale ridotta (patch_width, qui fissato a 2).
        
        Per ottenere una patch solo lungo la dimensione temporale,
        impostiamo lo stride verticale pari a patch_height.
        """
        super(Patches, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        # Impostiamo kernel_size=(patch_height, patch_width) e stride=(patch_height, 1)
        self.unfold = nn.Unfold(kernel_size=(patch_height, patch_width), stride=(patch_height, 1))

    def forward(self, x):
        # x: (batch, channels, height, width)
        # L'output di unfold ha forma (batch, C*patch_height*patch_width, L) dove L = numero di patch
        patches = self.unfold(x)  
        # Trasponiamo per avere (batch, num_patches, patch_dim)
        patches = patches.transpose(1, 2)
        return patches

############################################
# Modulo per il patch encoding
############################################
class PatchEncoder(nn.Module):
    def __init__(self, num_patches, patch_dim, projection_dim):
        """
        Proietta ciascuna patch (flattened) in uno spazio a dimensione projection_dim
        e aggiunge un embedding posizionale.
        """
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(patch_dim, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, x):
        # x: (batch, num_patches, patch_dim)
        x = self.projection(x)  # -> (batch, num_patches, projection_dim)
        # Creiamo gli indici posizionali (0, 1, ..., num_patches-1)
        positions = torch.arange(self.num_patches, device=x.device)
        x = x + self.position_embedding(positions)
        return x

############################################
# Un singolo Transformer Layer
############################################
class TransformerLayer(nn.Module):
    def __init__(self, projection_dim, num_heads, transformer_units, dropout_rate=0.7, reg_rate=None):
        """
        Ogni layer applica:
          1. LayerNorm
          2. Multi-Head Attention (con dropout=0.5)
          3. Skip connection
          4. Nuovo LayerNorm
          5. MLP (con attivazione ELU e dropout)
          6. Skip connection
        """
        super(TransformerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(projection_dim, eps=1e-6)
        # Utilizziamo batch_first=True per lavorare con tensori di forma (batch, seq_len, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, dropout=0.5, batch_first=True)
        self.norm2 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.mlp = MLP(projection_dim, transformer_units, dropout_rate, reg_rate)

    def forward(self, x):
        # x: (batch, num_patches, projection_dim)
        x_norm = self.norm1(x)
        # MultiHeadAttention: dato che batch_first=True, input mantiene la forma (batch, seq_len, embed_dim)
        attn_output, attn_weights = self.attention(x_norm, x_norm, x_norm, need_weights=True)
        # Skip connection 1
        x = x + attn_output
        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        # Skip connection 2
        x = x + mlp_output
        return x, attn_weights

############################################
# Modello principale: ConTraNet
############################################
class ConTraNet(nn.Module):
    def __init__(self, nb_classes, Chans, Samples, dropoutRate, regRate,
                 kernLength, poolLength, numFilters, dropoutType,
                 projection_dim, transformer_layers, num_heads,
                 transformer_units, mlp_head_units, training=True):
        """
        Parametri (con riferimento al commento del codice Keras):
          - nb_classes: numero di classi da classificare.
          - Chans: numero di canali (ad es. elettrodi EEG).
          - Samples: lunghezza della sequenza.
          - dropoutRate: tasso di dropout per il blocco CNN (es. 0.5).
          - regRate: parametro per il max norm (da applicare manualmente se necessario).
          - kernLength: lunghezza del kernel per il primo layer convoluzionale.
          - poolLength: fattore di pooling lungo la dimensione temporale.
          - numFilters: numero di filtri nel blocco CNN.
          - dropoutType: 'SpatialDropout2D' oppure 'Dropout'.
          - projection_dim: dimensione di proiezione per l’encoder delle patch.
          - transformer_layers: numero di layer Transformer.
          - num_heads: numero di teste nell’attenzione multi-testa.
          - transformer_units: lista (o int) con il numero di unità del MLP all’interno del Transformer.
          - mlp_head_units: lista (o int) con il numero di unità per il MLP finale.
          - training: se False, il modello restituisce anche le attention scores.
        """
        super(ConTraNet, self).__init__()
        self.return_attention = not training  # se non in training, restituiremo anche le attention scores

        # --------------------------
        # Blocco CNN
        # --------------------------
        # In PyTorch, si assume l’input di forma (batch, 1, Chans, Samples)
        # Il layer Conv2d:
        #   - in_channels = 1 (dato che l’input ha 1 canale)
        #   - out_channels = numFilters
        #   - kernel_size = (1, kernLength) -> si opera solo lungo la dimensione temporale
        #   - padding: per avere “same” padding lungo la dimensione temporale, usiamo padding=(0, kernLength//2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=numFilters, 
                               kernel_size=(1, kernLength), 
                               padding=(0, kernLength // 2), bias=False)
        # BatchNorm (sui canali)
        self.batch_norm = nn.BatchNorm2d(numFilters)
        # Attivazione ELU
        self.elu = nn.ELU()
        # Average Pooling lungo la dimensione temporale (kernel size=(1, poolLength))
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, poolLength))
        # Scegliamo il dropout in base al parametro dropoutType
        if dropoutType == 'SpatialDropout2D':
            self.dropout = nn.Dropout2d(dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropout = nn.Dropout(dropoutRate)
        else:
            raise ValueError("dropoutType deve essere 'SpatialDropout2D' oppure 'Dropout'.")

        # Raggruppiamo il blocco CNN in un Sequential
        self.cnn = nn.Sequential(
            self.conv1,
            self.batch_norm,
            self.elu,
            self.avg_pool,
            self.dropout
        )

        # --------------------------
        # Patch Creation & Encoding
        # --------------------------
        # Dopo il blocco CNN, l’output ha forma: (batch, numFilters, Chans, new_samples)
        # Assumiamo che il pooling lungo Samples sia divisibile, dunque:
        new_samples = Samples // poolLength  
        # La creazione delle patch prevede:
        #   - patch_height = Chans (si preleva l'intera dimensione spaziale)
        #   - patch_width = 2 (finestra temporale ridotta)
        # Lo stride viene impostato in modo da avere una sola patch verticale:
        #   stride = (Chans, 1)
        self.patches = Patches(patch_height=Chans, patch_width=2)
        # Calcoliamo il numero di patch: lungo la dimensione temporale,
        #   num_patches = (new_samples - 2) // 1 + 1 = new_samples - 1
        num_patches = new_samples - 1
        # Dimensione di ciascuna patch: i dati estratti da nn.Unfold sono di dimensione:
        #   patch_dim = numFilters * patch_height * patch_width = numFilters * Chans * 2
        patch_dim = numFilters * Chans * 2
        self.patch_encoder = PatchEncoder(num_patches=num_patches, patch_dim=patch_dim, projection_dim=projection_dim)

        # --------------------------
        # Blocco Transformer
        # --------------------------
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(projection_dim, num_heads, transformer_units, dropout_rate=0.7, reg_rate=regRate)
            for _ in range(transformer_layers)
        ])

        # Dopo il Transformer, effettuiamo una normalizzazione (LayerNorm)
        self.representation_norm = nn.LayerNorm(projection_dim, eps=1e-6)
        # Dropout sul vettore rappresentativo
        self.representation_dropout = nn.Dropout(0.5)

        # --------------------------
        # MLP Head e Classificazione
        # --------------------------
        # Il tensore dopo il Transformer ha forma (batch, num_patches, projection_dim).
        # Lo appiattiamo in (batch, num_patches * projection_dim)
        flattened_dim = num_patches * projection_dim
        self.mlp_head = MLP(flattened_dim, mlp_head_units, dropout_rate=0.7, reg_rate=regRate)
        # Layer finale: proiezione al numero di classi
        # Applichiamo quindi una Dense con attivazione ELU seguita da Softmax
        # Se mlp_head_units è una lista, l’input per il layer finale sarà l’ultimo elemento.
        if isinstance(mlp_head_units, (list, tuple)):
            mlp_head_out = mlp_head_units[-1]
        else:
            mlp_head_out = mlp_head_units
        self.classifier_dense = nn.Linear(mlp_head_out, nb_classes)

    def forward(self, x):
        """
        x è atteso in forma (batch, 1, Chans, Samples)
        """
        # Blocco CNN
        # Output shape: (batch, numFilters, Chans, new_samples)
        x = self.cnn(x)

        # Creazione delle patch
        # L'output di self.patches: (batch, num_patches, patch_dim)
        patches = self.patches(x)
        # Encoding delle patch
        encoded_patches = self.patch_encoder(patches)  # -> (batch, num_patches, projection_dim)

        # Blocco Transformer
        attn_scores = None
        for transformer in self.transformer_layers:
            encoded_patches, attn = transformer(encoded_patches)
            attn_scores = attn  # si conserva l'output dell'ultimo layer

        # Creazione della rappresentazione
        representation = self.representation_norm(encoded_patches)  # (batch, num_patches, projection_dim)
        representation = representation.flatten(start_dim=1)  # (batch, num_patches * projection_dim)
        representation = self.representation_dropout(representation)

        # MLP Head
        features = self.mlp_head(representation)
        # Classificazione finale: applica una Dense seguita da ELU e Softmax
        logits = self.classifier_dense(features)
        logits = F.elu(logits)
        output = F.softmax(logits, dim=1)

        # Se non siamo in training, restituiamo anche le attention scores
        if self.return_attention:
            return output, attn_scores
        else:
            return output

############################################
# Esempio di scheduler per il learning rate
############################################
def step_decay(epoch):
    """
    Restituisce il tasso di apprendimento in base all'epoca.
    Nel codice Keras:
      - epoch < 50 -> 0.001
      - altrimenti -> 0.0001
    Qui definiamo una funzione lambda per LambdaLR.
    """
    if epoch < 50:
        return 1.0   # moltiplicatore 1.0 (se l’optimizer è stato inizializzato con lr=0.001)
    else:
        return 0.1   # moltiplicatore 0.1 (0.001 * 0.1 = 0.0001)

# Esempio di utilizzo:
if __name__ == '__main__':
    # Parametri di esempio (modifica secondo le tue esigenze)
    nb_classes = 4
    Chans = 22
    Samples = 1000
    dropoutRate = 0.5
    regRate = 0.25
    kernLength = 64
    poolLength = 8
    numFilters = 16
    dropoutType = 'SpatialDropout2D'  # oppure 'Dropout'
    projection_dim = 32
    transformer_layers = 1
    num_heads = 8
    transformer_units = [projection_dim * 2, projection_dim]
    mlp_head_units = [112]
    
    # Crea il modello (in training restituisce solo l'output, altrimenti anche le attention scores)
    model = ConTraNet(nb_classes, Chans, Samples, dropoutRate, regRate,
                        kernLength, poolLength, numFilters, dropoutType,
                        projection_dim, transformer_layers, num_heads,
                        transformer_units, mlp_head_units, training=True)
    
    # Stampa la sthans, Samples)
    x = torch.randn(8, 1, Chans, Samples)
    out = model(x)
    print("Output shape:", out.shape)
    
    # Esempio di configurazione dello scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=step_decay)
    
    # Durante l'addestramento, dopo ogni epoca esegui:
    # scheduler.step()
    
    # Ricorda: per applicare il max norm constraint, potresti definire una funzione
    # che itera sui parametri (ad es. dei layer Linear e Conv2d) e ne clampa la norma.
