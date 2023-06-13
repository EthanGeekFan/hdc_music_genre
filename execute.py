"""
This function is used only for executing other files
"""

from hdc import HDC, hdc_preproc
from nn_model import Model
import get_data
import os
from datasets import load_dataset, load_from_disk, Dataset
import torch
import matplotlib.pyplot as plt
from keras_model import train_keras_model
# from data_augmentation import add_noise

# get and preprocess GTZAN into HDC dataset
HDC_DIM = 1024
split = True
num_split_pieces = 10
if not split:
    hdc_ds_path = f'gtzan_hdc_standard_{HDC_DIM}'
else:
    hdc_ds_path = f'gtzan_hdc_split_{num_split_pieces}_{HDC_DIM}'
if not os.path.exists(hdc_ds_path):
    if split:
        dataset = get_data.split_pieces(num_split_pieces)
    else:
        dataset = get_data.standard()
    dataset.set_format('torch', columns=['features'])
    dataset = hdc_preproc(dataset, HDC_DIM)
    dataset.save_to_disk(hdc_ds_path)
else:
    dataset = load_from_disk(hdc_ds_path)

# get augmented data
# dataset = load_from_disk(hdc_ds_path)


dataset.set_format('torch', columns=['hdc', 'genre'])
print('This is the shape of the dataset:', dataset.shape)

# split the dataset into training and testing
dataset = dataset.train_test_split(test_size=0.1, seed=229, stratify_by_column='genre')
train = dataset['train']
test = dataset['test']

# train neural network
model = Model(HDC_DIM, 100, 10)
losses, accuracies, test_accuracies = model.train(train['hdc'], train['genre'], 
                                                    torch.optim.Adam(model.parameters(), lr=0.00001),
                                                    test['hdc'], test['genre'],
                                                    epochs=50)

# import pdb; pdb.set_trace()


# evaluate the model
# plt.subplot(1, 2, 1)
# plt.plot(losses)
# plt.title('Loss')
# plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Train')
plt.plot(test_accuracies, label='Test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('HDC-ANN Training')
plt.legend()
plt.savefig('loss_acc.png')
model.test(test['hdc'], test['genre'])
model.save('model.pt')