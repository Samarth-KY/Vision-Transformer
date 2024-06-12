# Visual Transformers Implementation

![Visual Transformers](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)
*Fig 1. Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." [Paper](https://arxiv.org/pdf/2010.11929.pdf)*

Transformers have been studied in the context of sequence-to-sequence modelling applications like natural language processing (NLP). Their superior performance compared to LSTM-based recurrent neural networks is thanks to their ability to model long sequences. Recently, transformers have been adapted to the [visual domain](https://arxiv.org/abs/2010.11929) and have demonstrated better performance compared to convolutional neural networks when applied to large-scale datasets. This is due to their ability ...

## Implementation Overview

This repository provides an implementation of the building blocks of visual transformers (LightViT). The notebook trains them on classification tasks using MNIST and Fashion-MNIST datasets.

### 1. Image Patches and Linear Mapping

#### A) Image Patches

Transformers were initially created to process sequential data. In the case of images, a sequence can be created through extracting patches. A crop window is used with a defined window height and width. The dimension of data is originally in the format of *(B,C,H,W)*, and when transformed into patches and then flattened, it becomes *(B, PxP, (HxC/P)x(WxC/P))*, where *B* is the batch size and *PxP* is the total number of patches in an image. In this example, you can set P=7. 

*Output*: A function that extracts image patches. The output format should have a shape of (B,49,16). The function will be used inside the *LightViT* class.

#### B) Linear Mapping

The input is mapped using a linear layer to an output with dimension *d* i.e. *(B, PxP, (HxC/P)x(WxC/P))* → *(B, PxP, d)*. The variable d can be freely chosen, but here it is set to 8. 

*Output*: A linear layer should be added inside the *LightViT* class with the correct input and output dimensions. The output from the linear layer should have a dimension of (B,49,8).

### 2. Insert Classifier Token and Positional Embeddings

#### A) Classifier Token

Besides the image patches, also known as tokens, an additional special token is appended to the input to capture desired information about other tokens to learn the task at hand. Later, this token will be used as input to the classifier to determine the class of the input image. Adding the token to the input is equivalent to concatenating a learnable parameter with a vector of the same dimension *d* to the image tokens. 

*Output*: A randomly initialized learnable parameter to be implemented inside the *LightViT* class. You can use the [PyTorch built-in function](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html) to create a PyTorch parameter.

#### B) Positional Embedding

To preserve the context of an image, positional embeddings are associated with each image patch. Positional embeddings encode the patch positions using sinusoidal waves, as described in the original transformer paper by [Vaswani et. al](https://arxiv.org/abs/1706.03762). You'll be required to implement a function that creates embeddings for each coordinate of every image patch. 

### 3. Encoder Block

![Transformer Encoder](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)
*Fig 2. Transformer Encoder. [Paper](https://arxiv.org/pdf/2010.11929.pdf)*

Implementing the main elements of an encoder block is a key part of this notebook. A single block contains layer normalization (LN), multi-head self-attention (MHSA), and a residual connection.  

#### A) Layer Normalization

[Layer normalization](https://arxiv.org/abs/1607.06450) normalizes an input across the layer dimension by subtracting the mean and dividing by the standard deviation. You can instantiate layer normalization with a dimension *d* using the [PyTorch built-in function](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).

#### B) MHSA

![Multi-Head Self Attention](https://production-media.paperswithcode.com/methods/multi-head-attention_l1A3G7a.png)
*Fig 3. Multi-Head Self Attention. [Paper](https://arxiv.org/pdf/1706.03762v5.pdf)*

The attention module derives an attention value by measuring similarity between one patch and other patches. An image patch with dimension *d* is linearly mapped to three vectors: query **q**, key **k**, and value **v**. To quantify attention for a single patch, the dot product is computed between its **q** and all of the **k** vectors and divided by the square root of the vector dimension i.e. *d* = 8. The result is passed through a softmax layer to get *attention features* and finally multiplied with *...

This process is repeated **N** times on each of the **H** sub-vectors of the 8-dimensional patch, where **N** is the total number of attention blocks. In this case, **N** = 2, hence, we have 2 sub-vectors, each of length 4. The first sub-vector is processed by the first head and the second sub-vector is processed by the second head, each head having distinct Q, K, and V mapping functions of size 4x4. 

For more information about MHSA, you may refer to this [post](https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/).

It is highly recommended to define a separate class for MHSA as it contains several operations.

#### C) Residual Connection

Residual connections (also known as skip connections) add the original input to the processed output by a network layer, e.g., encoder. They have proven to be useful in deep neural networks as they mitigate problems like exploding/vanishing gradients. In transformers, the residual connection adds the original input to the output from LN → MHSA. All of the previous operations could be implemented inside a separate encoder class.

The last part of an encoder is to insert another residual connection between the input to the encoder and the output from the encoder passed through another layer of LN → MLP. The MLP consists of 2 layers with hidden size 4 times larger than *d*.

### 4. Classification Head

The final part of implementing a transformer is adding a classification head to the model inside the *LightViT* class. You can use a simple linear classifier, i.e., a linear layer that accepts input of dimension *d* and outputs logits with dimension set to the number of classes for the classification problem at hand.

### Model Training

The notebook includes a standard script for training and testing the model. It is recommended to use the Adam optimizer with a 0.005 learning rate and train for 5 epochs.
