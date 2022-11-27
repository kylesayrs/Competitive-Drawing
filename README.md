# Competitive-Drawing
A game where players compete to draw differing prompts on a shared canvas, as judged by a computer vision model

<img src="repo_assets/clock_spider.gif" alt="Competitive Drawing Logo"/>

### TODO ###
* research
    * train model
        * retrain with pairs of classes
        * try sgd optimizer (better generalization)
        * automated training
    * ai opponent
        * backpropogate through strokes defined by points?
        * two losses: discriminator/stroke loss and prediction score loss
        * self play
* game design
    * win condition
    * ramping up/down distance
    * showing grad cam power up?
    * erasing turn
    * cheat detection
* infrastructure
    * online multiplayer
    * protect api endpoints
    * dedicated inference instance
* ui
    * target selector for local play
    * add wave effect to distance indicator
    * option to expand preview
    * more appealing end turn button
    * better home screen button
