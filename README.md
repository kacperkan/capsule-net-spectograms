# Spectogram classification

## Description
This project includes all the code used during a private kaggle competition. This solution got 7th place on both public and private leaderboard.

## Requirements

- python 3.6
- pytorch 1.0
- tensorboardX

## Best solution description

1. Spectograms are split to 32 x 32 parts and the task is to classify which spectogram the part belongs to (there 3 parts).
2. Data is normalised to range [0, 1]
3. Achitecture consists of convolutional feature extractor and capsulenet head optimised according to this [article](https://arxiv.org/pdf/1806.07416.pdf).

## Additional info

For additional information / used data / trained models - send me a private message or an e-mail at kacper1095@gmail.com.

