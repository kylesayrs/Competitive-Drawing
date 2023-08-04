# Train #

## Classifier ##
This folder contains code to train a standard classifier trained on image label pairs. The images are merged with cutmix to simulate the "frankein" images likely to be seen during deployment and the model has logit norm applied in order to reduce over-confidence.

## Contrastive Learning ##
This folder contains code to train image and class encoders, similar to OpenAI's [CLIP](https://openai.com/research/clip).
