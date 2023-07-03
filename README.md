# eBISS 2023 Demo applications

In the context of:

## Leveraging HPC techniques for data analytics

## Repository content

```raw
.
├── custom-data
├── dc-pycompss-knn
│   └── model
├── dc-pytorch
│   └── model
├── pytorch-mnist
├── sklearn-digits
└── wordcount
    └── model
```

- `custom-data`: Contains images and texts used as sample datasets for the different applications in this repository.
- `dc-pycompss-knn`: A digit recognition training and inference done with PyCOMPS and dataClay (underlying implementation uses _k_-Nearest Neighbors and scikit-learn).
- `dc-pytorch`: A digit recognition training and inference done with dataClay. The implementation is done with PyTorch and a Convolutional Neural Network.
- `pytorch-mnist`: A digit recognition training and inference done with PyTorch. This is used as the reference implementation, to check correctness and behavior of the implementation. It will not be used, but it is left here for future reference.
- `sklearn-digits`: A digit recognition training and inference done with scikit-learn. This is used as the reference implementation, to check correctness and behavior of the implementation. It will not be used, but it is left here for future reference.
- `wordcount`: A Word Count application for demonstrating basic dataClay usage.
