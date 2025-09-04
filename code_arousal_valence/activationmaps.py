import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention_conformer(model, input_tensor, transformer_block_index=-1):
    """
    Esegue una forward pass e visualizza la mappa di attenzione media (sulle teste)
    del blocco Transformer indicato (default: ultimo blocco).

    Args:
        model (nn.Module): Modello Conformer.
        input_tensor (torch.Tensor): Input per il modello (es. shape [batch, n_chan, tempo]).
        transformer_block_index (int): Indice del blocco Transformer da cui prelevare la mappa.
    """
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)  # eseguiamo la forward per calcolare e salvare le mappe
        # Estraiamo il blocco transformer desiderato
        transformer_block = model.Attention.layers[transformer_block_index]
        # La mappa di attenzione salvata ha forma [batch, num_heads, query_length, key_length]
        att_map = transformer_block.attention.last_att  
        
    if att_map is None:
        print("Attenzione: la mappa di attenzione non è stata catturata.")
        return

    # Per visualizzare, ad esempio, la mappa media sulle teste del primo campione del batch
    att_avg = att_map[0].mean(dim=0)  # [query_length, key_length]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(att_avg.numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.title("Mappa di Attenzione Media (ultimo blocco)")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.colorbar()
    plt.show()


def visualize_attention_eegvit(model, input_tensor, layer_idx=0, head_idx=None):
    """
    Esegue una forward pass sul modello e visualizza la mappa di attenzione del layer specificato.
    
    Args:
        model (nn.Module): Il modello EEGViT.
        input_tensor (torch.Tensor): Input per il modello (es. shape [batch, num_chan, num_time]).
        layer_idx (int): Indice del layer Transformer da cui estrarre la mappa (default=0, primo layer).
        head_idx (int or None): Indice della testa da visualizzare. Se None, viene visualizzata la media sulle teste.
    """
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    # Recupera il modulo Attention dal layer Transformer indicato:
    # Ogni layer è una lista [PreNorm(Attention), PreNorm(FeedForward)]
    attn_module = model.transformer.layers[layer_idx][0].fn  # PreNorm wrap dell'Attention
    att_map = attn_module.last_att  # shape: [batch, heads, tokens, tokens]
    
    if att_map is None:
        print("Nessuna mappa di attenzione trovata!")
        return

    # Seleziona il primo campione del batch per la visualizzazione
    sample_att = att_map[0]  # shape: [heads, tokens, tokens]
    if head_idx is None:
        # Media sulle teste
        att_avg = sample_att.mean(dim=0)
        data_to_plot = att_avg
        title = f"Layer {layer_idx} - Media su tutte le teste"
    else:
        data_to_plot = sample_att[head_idx]
        title = f"Layer {layer_idx} - Testa {head_idx}"
    
    plt.figure(figsize=(6, 6))
    plt.imshow(data_to_plot.numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.colorbar()
    plt.show()


def visualize_attention_deformer(model, input_tensor, layer_idx=0, head_idx=None):
    """
    Visualizza la mappa di attenzione del layer Transformer specificato all'interno del modello Deformer.

    Args:
        model (nn.Module): Il modello Deformer.
        input_tensor (torch.Tensor): Input di forma (batch, num_chan, num_time).
        layer_idx (int): Indice del layer Transformer da cui estrarre la mappa (default=0).
        head_idx (int or None): Indice della testa da visualizzare; se None viene usata la media sulle teste.
    """
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    # Ogni layer nel Transformer è una lista: [Attention, FeedForward, cnn_block]
    att_module = model.transformer.layers[layer_idx][0]
    if not hasattr(att_module, "last_att") or att_module.last_att is None:
        print("Nessuna mappa di attenzione trovata in questo layer.")
        return
    att_map = att_module.last_att  # forma: [batch, heads, tokens, tokens]
    # Seleziona il primo campione del batch per visualizzare
    sample_att = att_map[0]  # forma: [heads, tokens, tokens]
    if head_idx is None:
        # Media sulle teste
        att_to_plot = sample_att.mean(dim=0)
        title = f"Layer {layer_idx} - Media su tutte le teste"
    else:
        att_to_plot = sample_att[head_idx]
        title = f"Layer {layer_idx} - Testa {head_idx}"
    plt.figure(figsize=(6,6))
    plt.imshow(att_to_plot.numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.colorbar()
    plt.show()

def visualize_attention_contranet(model, input_tensor, layer_idx=-1):
    """
    Esegue una forward pass sull'input e visualizza la mappa di attenzione
    dell'ULTIMO blocco transformer (o, se si vuole, di uno specifico layer modificando il codice).
    
    Args:
        model (nn.Module): Il modello ConTraNet.
        input_tensor (torch.Tensor): Input con forma (batch, channels, samples).
        layer_idx: qui, lasciamo -1 per indicare l'ultimo blocco (poiché in questo modello,
                   i pesi di attenzione restituiti sono quelli dell'ultimo transformer block).
    """
    model.eval()
    # Impostiamo il flag in modo che il forward restituisca anche i pesi di attenzione
    model.training_flag = False
    with torch.no_grad():
        outputs = model(input_tensor)
    # outputs è una tupla (logits, attn_weights)
    logits, attn_weights = outputs
    # attn_weights ha forma (batch, num_patches, num_patches)
    sample_att = attn_weights[0]  # prendiamo il primo campione del batch
    plt.figure(figsize=(8, 8))
    plt.imshow(sample_att.cpu().numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.title("Mappa di Attenzione - Ultimo Transformer Block")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.colorbar()
    plt.show()