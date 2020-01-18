# Credicxo_assignment

The dataset given contains details about organic chemical compounds including their chemical features, isomeric conformation, names and the classes in which they are classified. The compounds are classified as either ‘Musk’ i.e '1' or ‘Non-Musk’ i.e '0' compounds. 

# Input:
170 coloumns with 6598 rows of data

# Pre-processing: 
1. Out of 170 coloumns 166 were relevant for output, so they were seperated from the input and taken in an array.
2. The last coloumn is taken as output array.
3. The datasets are then split in train and test set in 80:20 ratio respectively.
4. Then the input dataset was Scaled(Normalized) for better performance by neural network.

# The Training:
1. The Sequential neural network of Multi-Layer perceptron is made with each layer uniformly initialized with ReLu as activation function.
2. The last layer of the network has 1 output with sigmoid as activation function.
3. The checkpoints are made which monitor 'val_loss' and saves the weight which is better than previous ones and the logs are also made with Tensorboard.
4. With 30 epochs and batch size of 300 the training is started on 5278 samples and validation on 1320 samples.

# The Observations:
1. Validation loss: 0.0129 ( <1%)
2. Validation accuracy: 0.9985 (99.85%)
3. Precision: 0.99 ( >99%)
4. Recall: 1.0 (100%)
5. F1 score: 0.9950 (99.5%)
