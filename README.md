# DERFormer: Driver Emotion Recognition Transformer via Dual Feature Enhancement for Driver Monitoring System

This repository is the official implementation of the paper **DERFormer: Driver Emotion Recognition Transformer via Dual Feature Enhancement for Driver Monitoring System**.

## Prepare

Download pretrained weights (`ir101.pt`, `pe.pt` and `vit.pt`) from https://huggingface.co/TTJiang/DERFormer and place them into `./pretrained_weights`

## Run

```
python preprocess.py --cache-root Path/to/Cached/Data --mli-der-root Path/to/MLI-DER/Dataset --kmu-fed-root Path/to/KMU-FED/Dataset

python train.py --cache-root Path/to/Cached/Data/MLI-DER

python train.py --cache-root Path/to/Cached/Data/KMU-FED

```

## License

MIT