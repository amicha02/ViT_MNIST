# Vision Transformer (ViT) 
The ___init___ method is used to initialize the attributes of the class, which can include hyperparameters, learnable parameters, and model components. The hyperparameters, in particular, are values that you set to configure the model before training, and they influence the model's structure and behavior.
## 1. Architecture Hyperparameters
- `image_size`: The size of input images in pixels. It must be divisible by `patch_size`.
- `patch_size`: The size of non-overlapping image patches. The image is divided into patches of this size.
- `num_classes`: The number of output classes for the classification task.
- `dim`: Dimensionality of the token embeddings. Determines the model's capacity to capture features.
- `depth`: The number of transformer blocks in the model. More blocks allow for more complex feature extraction.
- `heads`: The number of attention heads in the multi-head attention mechanism. Each head attends to different aspects of the input.
- `mlp_dim`: Dimensionality of the feedforward network inside the transformer blocks.
- `channels`: The number of input channels in the image (e.g., 1 for grayscale, 3 for RGB).

Learning rate, batch size, and training epochs are common training hyperparameters not explicitly shown in the ViT class. 


### Positional Embeddings
Positional embeddings are used to provide spatial information to the model. There are both learnable parameters (along with the cls_token)
- `patch_to_embedding`: Linear layer to transform patch features into embedding space.
- `pos_embedding`: Learnable positional embeddings added to the patch embeddings.

### Transformer Layers
- `Transformer`: A stack of transformer blocks, each consisting of multi-head self-attention and feedforward layers.
- `Attention`: Multi-head self-attention mechanism with linear transformations for query, key, and value.

### Output Head
- `to_cls_token`: Identity operation, extracting the classification token from the output.
- `mlp_head`: A multi-layer perceptron (MLP) head for final classification, consisting of linear layers and GELU activation.

### Training and Inference
- Training typically involves optimizing the model's parameters using a suitable loss function and backpropagation.
- During inference, the model is used to predict the class of input images.

### Miscellaneous
- `patch_dim`: The dimensionality of each image patch feature, calculated based on `channels` and `patch_size`.
- `cls_token`: Learnable classification token used to aggregate information from different patches.

## ViT _init__ breakdown
- [nn.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html) \
    In PyTorch, when you're building a neural network, you often have some values (parameters) that the network needs to learn during training. These could be things like weights in the layers of your network.


    The Parameter class is a way to represent and manage these learnable parameters. It's like a special type of number that PyTorch knows it needs to adjust (or learn) as your neural network learns from data.
- [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)\
This applies linear transformation to the incoming data.
$$ y = {xA^T + b } $$


## ViT Forward breakdown