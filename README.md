## Closed-form Sample Probing for Learning Generative Models in Zero-shot Learning

The official repository for the [Closed-form Sample Probing for Learning Generative Models in Zero-shot Learning](https://openreview.net/forum?id=ljxWpdBl4V) paper published at ICLR 2022.


<p align="center"> <img src = "images/meta-genzsl-intro.png" width="700"> </p>
<p align="left"> Figure: Illustration of the proposed framework for the end-to-end sample probing of conditional generative models. </p>


## Data
Proposed data splits for all datasets can be found [here](https://drive.google.com/drive/folders/16Xk1eFSWjQTtuQivTogMmvL3P6F_084u).

## Results

<p align="left"> Table: Generalized zero-shot learning scores of sample probing with alternative closed-form models, based on TF-VAEGAN baseline. </p>
<p align="center"> <img src = "images/sample-probing-with-alternative-closed-form-models.png" width="800"> </p>

## Citation
If you find this code useful in your research, please consider citing as follows:
```
@inproceedings{
cetin2022closedform,
title={Closed-form Sample Probing for Learning Generative Models in Zero-shot Learning},
author={Samet Cetin and Orhun Bu{\u{g}}ra Baran and Ramazan Gokberk Cinbis},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=ljxWpdBl4V}
}
```
## Acknowledgements
The parts of the code related to generative model (TF-VAEGAN) training is taken/adapted from [tfvaegan](https://github.com/akshitac8/tfvaegan) repository.
