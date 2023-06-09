# A stronger implementation of IndexNet.

Yet another implementation of the paper [**Indices Matter: Learning to Index for Deep Image Matting**](https://arxiv.org/abs/1908.00672)

## Description

A stronger implementation of IndexNet.

## Requirements
#### Hardware:

GPU memory >= 10GB for inference on Adobe Composition-1K testing set.

#### Packages:

- torch >= 1.10
- numpy >= 1.16
- opencv-python >= 4.0

## Models
**The model can only be used and distributed for noncommercial purposes.** 

| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| IndexNet (Paper) | 12MiB | 45.8 | 13.0 | 25.9 | 43.7 |
| IndexNet (Ours) | 12MiB | 29.07 | 6.06 | 12.51 | 25.13 |

## Evaluation
We provide the script `eval.py`  for evaluation.



