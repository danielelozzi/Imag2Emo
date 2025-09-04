def contranet( nb_classes, Chans, Samples, dropoutRate, regRate,
              kernLength, poolLength, numFilters,
              dropoutType, projection_dim,
              transformer_layers, num_heads,
              transformer_units, mlp_head_units, training=True):
    import contranet_pytorch_backup

    """
    Creates a ConTraNet model.

    Args:
        nb_classes: Number of classes to categorize.
        Chans: Number of channels.
        Samples: Sequence length.
        dropoutRate: Dropout rate for CNN block.
        regRate: Regularization rate.
        kernLength: Kernel length for EEG (sampling frequency/2).
        poolLength: Pooling length.
        numFilters: Number of filters.
        dropoutType: Dropout type ('SpatialDropout2D' or 'Dropout').
        projection_dim: Projection dimension.
        transformer_layers: Number of transformer layers.
        num_heads: Number of heads in multi-head attention.
        transformer_units: Units in transformer block.
        mlp_head_units: Units in MLP head.
        training: Whether the model is in training mode.

    Returns:
        A ConTraNet model.
    """
    return contranet_pytorch_backup.ConTraNet(nb_classes, Chans, Samples, dropoutRate, regRate,
                                    kernLength, poolLength, numFilters,
                                    dropoutType, projection_dim,
                                    transformer_layers, num_heads,
                                    transformer_units, mlp_head_units, training)

def ertnet( nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8, heads=8,
           D=2, F2=16):
    import ERTnet_pytorch
    """
    Creates an ERTNet model.

    Args:
        nb_classes: Number of classes to categorize.
        Chans: Number of channels.
        Samples: Sequence length.
        dropoutRate: Dropout rate.
        kernLength: Kernel length.
        F1: Number of filters in the first convolutional layer.
        heads: Number of heads in multi-head attention.
        D: Depth multiplier.
        F2: Number of filters in the second convolutional layer.

    Returns:
        An ERTNet model.
    """
    return ERTnet_pytorch.ertnet_def(nb_classes, Chans, Samples,
                                dropoutRate, kernLength, F1, heads,
                                D, F2)

def eegdeformer( num_chan, num_time, temporal_kernel, num_kernel=64,
                num_classes=2, depth=4, heads=16,
                mlp_dim=16, dim_head=16, dropout=0.):
    import EEGDeformer

    """
    Creates an EEGDeformer model.

    Args:
        num_chan: Number of channels.
        num_time: Number of time points.
        temporal_kernel: Size of the temporal kernel.
        num_kernel: Number of kernels in the convolutional layers.
        num_classes: Number of classes to categorize.
        depth: Depth of the transformer.
        heads: Number of heads in multi-head attention.
        mlp_dim: Dimension of the MLP layers.
        dim_head: Dimension of the attention heads.
        dropout: Dropout rate.

    Returns:
        An EEGDeformer model.
    """
    return EEGDeformer.Deformer(num_chan=num_chan, num_time=num_time, temporal_kernel=temporal_kernel, num_kernel=num_kernel,
                                num_classes=num_classes, depth=depth, heads=heads,
                                mlp_dim=mlp_dim, dim_head=dim_head, dropout=dropout)

def eegnet( nChan, nTime, nClass=2,
           dropoutP=0.25, F1=8, D=2,
           C1=64):
    import EEGNet

    """
    Creates an EEGNet model.

    Args:
        nChan: Number of channels.
        nTime: Number of time points.
        nClass: Number of classes to categorize.
        dropoutP: Dropout rate.
        F1: Number of filters in the first convolutional layer.
        D: Depth multiplier.
        C1: Size of the first convolutional kernel.

    Returns:
        An EEGNet model.
    """
    return EEGNet.eegNet(nChan, nTime, nClass,
                        dropoutP, F1, D,
                        C1)

def eegvit( num_chan, num_time, num_patches, num_classes, dim=32, depth=4, heads=16, mlp_dim=64, pool='cls', dim_head=64, dropout=0.1, emb_dropout=0.1):
    import EEGViT

    """
    Creates an EEGViT model.

    Args:
        num_chan: Number of channels.
        num_time: Number of time points.
        num_patches: Number of patches to divide the input into.
        num_classes: Number of classes to categorize.
        dim: Embedding dimension.
        depth: Depth of the transformer.
        heads: Number of heads in multi-head attention.
        mlp_dim: Dimension of the MLP layers.
        pool: Pooling method ('cls' or 'mean').
        dim_head: Dimension of the attention heads.
        dropout: Dropout rate.
        emb_dropout: Embedding dropout rate.

    Returns:
        An EEGViT model.
    """
    return EEGViT.EEGViT(num_chan=num_chan, num_time=num_time, num_patches=num_patches,
                        num_classes=num_classes, dim=dim, depth=depth, heads=heads,
                        mlp_dim=mlp_dim, pool=pool, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)

# Example usage
#models = eegmodels()
#model = models.contranet(nb_classes=2, Chans=64, Samples=128, dropoutRate=0.5, regRate=0.25,
#                          kernLength=64, poolLength=8, numFilters=16,
#                          dropoutType='SpatialDropout2D', projection_dim=32,
#                          transformer_layers=1, num_heads=8,
#                          transformer_units=8, mlp_head_units=[112])

def conformer(n_chan, n_classes, n_times=None, n_patches=None, emb_size=40, depth=6, n_hidden=2440,
              n_heads=10, patch_size=25, pool_size=75, pool_stride=15, num_shallow_filters=40,
              custom_n_hidden=None, **kwargs):

    import conformer_pytorch
    """
    Crea un modello Conformer generalizzato per il decoding EEG.
    
    Args:
        n_chan (int): Numero di canali EEG.
        n_classes (int): Numero di classi per la classificazione.
        n_times (int, optional): Numero di punti temporali (non usato direttamente nel modello).
        n_patches (int, optional): Numero di patch (non usato direttamente, poiché il patching è gestito dal pooling).
        emb_size (int): Dimensione dell'embedding. Defaults a 40.
        depth (int): Profondità del Transformer encoder. Defaults a 6.
        n_hidden (int): Numero di unità nascoste nell'head di classificazione (usato se custom_n_hidden è None). Defaults a 2440.
        n_heads (int): Numero di teste nell'attenzione multi-head. Defaults a 10.
        patch_size (int): Dimensione del kernel per la prima conv in PatchEmbedding. Defaults a 25.
        pool_size (int): Dimensione della finestra di pooling in PatchEmbedding. Defaults a 75.
        pool_stride (int): Stride del pooling in PatchEmbedding. Defaults a 15.
        num_shallow_filters (int): Numero di filtri nella parte convoluzionale di PatchEmbedding. Defaults a 40.
        custom_n_hidden (int, optional): Se fornito, sovrascrive n_hidden.
        **kwargs: Argomenti aggiuntivi.
    
    Returns:
        torch.nn.Module: Un modello Conformer.
    """
    if custom_n_hidden is not None:
        n_hidden = custom_n_hidden

    return conformer_pytorch.Conformer(
        n_chan=n_chan,
        n_classes=n_classes,
        emb_size=emb_size,
        depth=depth,
        n_hidden=n_hidden,
        n_heads=n_heads,
        patch_size=patch_size,
        pool_size=pool_size,
        pool_stride=pool_stride,
        num_shallow_filters=num_shallow_filters,
        **kwargs
    )
