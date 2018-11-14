# Universal-Transformer-Pytorch
Simple and self-contained implementation of the [Universal Transformer](https://arxiv.org/abs/1807.03819) (Dehghani, 2018) in Pytorch. Please open issues if you find bugs, and send pull request if you want to contribuite. 

![](file.gif)
GIF taken from: [https://twitter.com/OriolVinyalsML/status/1017523208059260929](https://twitter.com/OriolVinyalsML/status/1017523208059260929)

## Universal Transformer 
The basic Transformer model has been taken from [https://github.com/kolloldas/torchnlp](https://github.com/kolloldas/torchnlp). For now it has been implemented:

- Universal Transformer Encoder Decoder, with position and time embeddings.
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) (Graves, 2016) as describe in Universal Transformer paper. 
- Universal Transformer for bAbI data.  
 
## Dependendency
```
python3
pytorch 0.4
torchtext
argparse
```
## How to run
To run standard Universal Transformer on bAbI run:
```
python main.py --task 1
```
To run Adaptive Computation Time: 
```
python main.py --task 1 --act
```

## TODO
- Run task 1-20 and report performance
- Visualize ACT on different tasks 
