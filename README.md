# EfficientMorph

This repository contains the official implementations of

**[EfficientMorph: Parameter-Efficient Transformer-Based Architecture for 3D Image Registration](https://arxiv.org/abs/2403.11026) (WACV-2025 oral presentation)**;

## Train and Test

The default configuration trains on EfficientMOrph-23 variants with the patch size of 2. To change the variant and patch size use different configurations from the ```models/configs.py``` file.

```
# Train on OASIS
python train_EfficientMorph.py --dataset_name OASIS --evaluation True
# Train on IXI
python train_EfficientMorph.py --dataset_name IXI --evaluation True 
# Train on Remind2Reg
python train_EfficientMorph.py --dataset_name Remind2Reg
```

If you find the code helpful, please consider citing our work:

```
@article{aziz2024efficientmorph,
  title={EfficientMorph: Parameter-Efficient Transformer-Based Architecture for 3D Image Registration},
  author={Aziz, Abu Zahid Bin and Karanam, Mokshagna Sai Teja and Kataria, Tushar and Elhabian, Shireen Y},
  journal={arXiv preprint arXiv:2403.11026},
  year={2024}
}
```

## Acknowledgments

We would like to acknowledge the [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) project, from which we have adopted the code used in our work.