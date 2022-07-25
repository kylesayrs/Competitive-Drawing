# Competitive-Drawing
A game where players compete to draw differing prompts on a shared canvas, as judged by a computer vision model

TODO
* name?
* retrain model
    * -90 to 90 rotations (no upside down)
    * slight crop perturbations
    * investigate exactly how data was created so we can match
    * retrain with more classes
* server side inference
    * at the end mouse out to validate data and correct results
* game mechanics
    * count pixels changed / stroke length
    * multiplayer sockets
    * erasing turn for loser if winning by more than 30%
* better ui
    * canvas on left
    * preview bottom right
    * plot vertically, show percentages at top of bars
    * only show relevant classes
        * me is on left, opponent is on right
* bug fixing
    * canvas when scrolled
