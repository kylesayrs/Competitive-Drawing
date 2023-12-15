# Train #

## Classifier ##
This folder contains code to train a standard classifier trained on image label pairs. The images are merged with cutmix to simulate the "frankein" images likely to be seen during deployment and the model has logit norm applied in order to reduce over-confidence.

## Contrastive Learning ##
This folder contains experimental code to train image and class encoders for latent space learning

## Reinforcement Learning ##
This folder contains experimental code to train a stroke agent. Due to the complexity of the input space of images and the sparsity of rewards, sample efficiency is low. Techniques such as world modeling may help to increase sample efficiency to the point of being trainable with fewer computational resources.
