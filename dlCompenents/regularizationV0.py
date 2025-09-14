##Regularization methods in deep learning are techniques used to prevent overfitting and improve the generalization ability of a model. Overfitting occurs when a model learns the training data too well, including its noise and specific patterns, leading to poor performance on unseen data. Regularization aims to add constraints or penalties to the model to make it simpler, more robust, and less prone to memorizing the training set.
##
##Why Regularize?
##Deep learning models, especially those with many parameters (like large neural networks), have a high capacity to learn. Without regularization, they can easily overfit if the training data isn't perfectly representative or sufficiently large.
##
##Here are the main regularization methods, with examples:
##
###-------------------------------------------------------------------
### 1. L1 and L2 Regularization (Weight Decay)
###-------------------------------------------------------------------
##
##These methods add a penalty term to the loss function during training, encouraging the model's weights to be smaller. Smaller weights generally lead to simpler models that are less sensitive to specific input features.
##
##1. L1 Regularization (Lasso Regularization):
##Penalty Term: Adds the sum of the absolute values of the model's weights (lambda sum∣w_i∣).
##Effect: Encourages sparsity in the weights, meaning it can drive some weights to exactly zero. This effectively performs feature selection, as features with zero-weighted connections are "removed" from the model.
##Example:
##If your original loss function is Mean Squared Error (MSE), with L1 regularization, the new loss becomes:
##Loss_new=MSE(y,haty)+lambda * sum_i∣w_i∣
##Here,
##lambda (lambda) is the regularization strength, a hyperparameter you tune.
##
##2. L2 Regularization (Ridge Regularization / Weight Decay):
##Penalty Term: Adds the sum of the squared values of the model's weights (lambda * sum w_i^2). This is the most common form of weight decay.
##Effect: Encourages weights to be small but rarely drives them to exactly zero. It spreads the importance more evenly across features.
##
##Example:
##With L2 regularization, the new loss becomes:
##Loss_new=MSE(y, haty) + lambda sum_i (w_i^2)
##Most deep learning frameworks (like PyTorch and TensorFlow) implement L2 regularization by adding weight_decay to the optimizer, which achieves the same effect.
##PyTorch Example (using weight_decay in optimizer):

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
model = nn.Linear(10, 1)

# L2 regularization (weight decay) is added directly to the optimizer
# A common value for weight_decay is 1e-4 or 1e-5
optimizer_l2 = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# For L1 regularization, you'd typically implement it manually in the loss calculation:
# l1_lambda = 0.001
# l1_norm = sum(p.abs().sum() for p in model.parameters())
# loss = original_loss + l1_lambda * l1_norm


#-------------------------------------------------------------------
#2. Dropout
#-------------------------------------------------------------------

##What it does: During training, randomly sets a fraction of the neurons' outputs to zero at each forward pass. This means that a different "thinned" network is used for each mini-batch.
##Purpose: Prevents co-adaptation of neurons (where neurons rely too heavily on specific other neurons). It forces the network to learn more robust and redundant representations, as no single neuron can be solely relied upon.
##How it works: You specify a dropout probability (e.g., 0.5), meaning 50% of the neurons in that layer will be randomly dropped out. During inference (testing), dropout is turned off, and the activations are scaled by the dropout probability to maintain the expected output magnitude from training.
##Example: Imagine a team where, for each task, a random subset of team members are "absent." This forces the remaining members to learn to perform well even without their usual teammates, making the whole team more resilient.


import torch.nn as nn

class MyModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(p=0.5) # Drop 50% of neurons
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.3) # Drop 30% of neurons
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        x = self.dropout1(x) # Apply dropout after activation (or before, depends on specific architecture)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# During training: model.train() enables dropout
# During evaluation: model.eval() disables dropout

#-------------------------------------------------------------------
# 3. Early Stopping
#-------------------------------------------------------------------

##What it does: Monitors the model's performance on a separate validation set during training. When the performance on the validation set starts to degrade (e.g., validation loss starts increasing), training is stopped, and the model weights from the best-performing epoch on the validation set are restored.
##Purpose: Prevents overfitting by halting the training process at the optimal point before the model begins to memorize the training data.
##How it works: You define a "patience" parameter. If the validation loss doesn't improve for patience number of epochs, training stops.
##Example: You're studying for an exam. Early stopping is like realizing you're starting to just memorize answers instead of understanding concepts (validation score drops), so you stop studying to prevent "overfitting" to your practice questions.
##Conceptual Example:


# During training loop:
# best_val_loss = float('inf')
# patience_counter = 0
# num_epochs = 100

# for epoch in range(num_epochs):
#     # ... (train model for one epoch) ...
#     # Calculate validation_loss
#     
#     if validation_loss < best_val_loss:
#         best_val_loss = validation_loss
#         patience_counter = 0
#         torch.save(model.state_dict(), 'best_model.pth') # Save best model
#     else:
#         patience_counter += 1
#         if patience_counter >= max_patience: # max_patience is your hyperparameter
#             print("Early stopping!")
#             break
#
# model.load_state_dict(torch.load('best_model.pth')) # Load the best model


#-------------------------------------------------------------------
#4. Data Augmentation
#-------------------------------------------------------------------

##What it does: Artificially increases the size and diversity of the training dataset by creating modified versions of existing data.
##Purpose: Exposes the model to more variations of the input data, making it more robust and less likely to overfit to specific examples. It's especially effective when the original dataset is small.
##How it works: For image data, common augmentations include: rotations, flips, scaling, cropping, translations, changes in brightness/contrast, color jittering. For text, it could be synonym replacement, random insertion/deletion of words.
##Example: To train a model to recognize cats, you can take an image of a cat and create new training examples by rotating it, flipping it horizontally, cropping different parts of it, or slightly changing its colors. The model then learns that a cat is a cat, regardless of these minor variations.
##PyTorch Example (for Images using torchvision.transforms):


from torchvision import transforms
from torchvision.datasets import CIFAR10

# Define augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),   # Randomly crop and resize
    transforms.RandomHorizontalFlip(), # Randomly flip image horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly change brightness/contrast
    transforms.ToTensor(),              # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize pixel values
])

# Apply transform to dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
# DataLoader will then feed augmented images during training

#-------------------------------------------------------------------
# 5. Batch Normalization
#-------------------------------------------------------------------

##What it does: Normalizes the inputs to each layer within a mini-batch by re-centering them to zero mean and unit variance.
##Purpose (Primary): To stabilize and speed up training by reducing internal covariate shift (the change in the distribution of network activations due to weight updates in preceding layers).
##Purpose (Regularization Effect): Although its primary goal isn't regularization, Batch Normalization has a regularizing effect. The mini-batch statistics used for normalization introduce a slight amount of noise into the network's activations. This noise discourages co-adaptation of neurons, similar to dropout, and can sometimes reduce the need for other regularization methods.
##Example: Imagine an assembly line where each station (layer) has its incoming parts (activations) normalized to a consistent size and position. This makes the work of subsequent stations much easier and more predictable. The slight variations in the batch statistics during training act as a form of "stochasticity."


import torch.nn as nn

class MyModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256) # Normalizes 256 features
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        x = self.fc3(x)
        return x


#-------------------------------------------------------------------
# 6. Label Smoothing
#-------------------------------------------------------------------

##What it does: Modifies "hard" one-hot encoded target labels (e.g., [0, 0, 1, 0]) into "soft" targets during training. Instead of assigning a probability of 1 to the correct class and 0 to others, it gives the correct class a slightly lower probability and distributes a small amount of probability mass to the incorrect classes.
##Purpose: Prevents the model from becoming overly confident in its predictions, which can lead to overfitting and poor generalization, especially when training data labels might be noisy or imprecise. It encourages the model to be less extreme in its logits for the true class.
##How it works: For a true class k in a K-class classification problem, the smoothed target probability y′_k is calculated as:
##y′_k=(1−alpha) cdot y_k + frac alpha * K
##where y_k is the original one-hot target (1 for true class, 0 otherwise), and alpha is the smoothing parameter (a small value like 0.1).
##Example: If an image is 100% a "cat," label smoothing might tell the model to aim for 90% "cat" and 10% "other animals." This prevents the model from pushing its confidence to extreme values (like 1.0) and makes it more robust to small variations or errors in the data.


#-------------------------------------------------------------------
# 7. Mixup and CutMix
#-------------------------------------------------------------------

##These are advanced data augmentation techniques that go beyond simple transformations by creating new training samples through combinations of existing ones.
##
##1. Mixup:
##What it does: Creates new training samples by linearly interpolating pairs of training examples and their labels.
##How it works: For two samples (x_i,y_i) and (x_j,y_j), a new sample (
##tildex,tildey) is generated as:
##tildex = lambda x_i + (1−lambda) x_j
##tildey = lambda y_i + (1−lambda) y_j
##where lambda is a random value drawn from a Beta distribution (e.g., Beta(alpha, alpha)).
##
##Purpose: Encourages the model to behave linearly in between training examples, promoting smoother decision boundaries and improving generalization. It also effectively expands the training manifold.
##Example: If you have an image of a cat and an image of a dog, Mixup creates a "blended" image that is partly cat and partly dog, with a blended label (e.g., 70% cat, 30% dog).
##
##2. CutMix:
##
##What it does: Similar to Mixup but creates new samples by cutting a patch from one image and pasting it onto another image, while mixing the labels proportionally to the area of the patches.
##How it works: A random bounding box is selected. The region inside this box in image A is replaced by the corresponding region from image B. The label is then a weighted average of the original labels, where the weights are based on the proportion of pixels from each image.
##Purpose: Encourages the model to focus on more global contextual information rather than relying on local textures. It also acts as a strong data augmentation and regularization.
##
##Example: Takes a cat image, cuts a square from it, and pastes that square onto a dog image. The new image's label would be, for example, 80% dog and 20% cat if 20% of the dog image's pixels were replaced by cat pixels.
##These are the most common and effective regularization methods in deep learning. Often, a combination of these techniques (e.g., L2 regularization, dropout, and data augmentation) is used to achieve the best results.