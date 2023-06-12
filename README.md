# Music Genre Classification with Hyperdimensional Computing

## Run model

```bash
conda env create -f environment.yml
conda activate music
python final.py
```

`pre.py` contains code for preprocessing the data, including cutting the audios to the same length or restructuring the data to be queried by the training code.

`final.py` contains code for training the model and testing it on the test set.

The model loads GTZAN dataset from HuggingFace Datasets. The dataset is downloaded automatically when running the code for the first time.

This HDC classification model achieves an accuracy of 0.65 on the test set and is comparable to the accuracy of a LSTM model (0.68) trained on the same dataset.

## Cite

```bibtex
@article{music_genre_classification,
  author = {Yifan Yang, Jerry Tang},
  title = {Music Genre Classification with Hyperdimensional Computing},
  year = {2023},
  publisher = {Stanford CS229},
  journal = {Stanford CS229 Project Gallary},
}
```