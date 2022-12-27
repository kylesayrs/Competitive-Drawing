# Competitive-Drawing
A game where players compete to draw differing prompts on a shared canvas, as judged by a computer vision model

<img src="repo_assets/clock_spider.gif" alt="Competitive Drawing Logo"/>


## Differentiable Graphics for AI Opponent ##
This work builds upon [Tzu-Mao Li 2020 "Differentiable Vector Graphics Rasterization for Editing and Learning"](https://people.csail.mit.edu/tzumao/diffvg/). My purpose is to build an AI opponent that draws a curve on the given canvas that optimizes the AI's score while being constrained by the borders of the image and a maximum curve length.

### Problem Statement ###
Fundamentally, I want to building an AI that, given a base image, produces changes that increase its score as judged by a classifier. A naive attempt to build a model that can do this might involve encoding the original image and producing a learned mask that, when added to the original image, produces a high score from the classifier. However, previous work by [Dawei Zhou et al.](https://arxiv.org/abs/2109.09901) has shown that this kind of architecture produces masks that better resemble random noise rather than deliberate, discrete strokes like you'd expect from a human.

<img src="./repo_assets/adversarial_noise.png" alt="Adversarial Noise" href="https://towardsdatascience.com/breaking-neural-networks-with-adversarial-attacks-f4290a9a45aa"/>  

My approach is to learn a curve which resembles an actual human pen stroke while still increasing the AI's score. While this method will not produce scores that are higher than pixel perfect adversarial noise, it will produce a AI opponent that is both challenging and fun to play against.

### Bézier Curve Differentiation ###
Curves, unlike individual pixels values, are not obviously differentiable. To make curves differentiable, we can take advantage of an existing graphics technique called anti-aliasing which essentially adds blur around a line which is both pleasing to the eye and much more differentiable than binary valued pixels.

In order to implement anti-aliasing of a vector curve, we must compute the shortest distance of each pixel to the curve in order to determine its brightness value. In the special case of a straight line, there is a close form solution that involves projecting the point onto the line and computing the distance from that projection point to our original point.

<img src="./repo_assets/distance_from_point_to_line.jpg" alt="Distance From Point to Line" href="https://www.chilimath.com/lessons/advanced-algebra/distance-between-point-and-line-formula/"/>

In the case of Bezier curves however, no closed form solution for projection exists. This is noted by [Tzu-Mao et al.](https://people.csail.mit.edu/tzumao/diffvg/) The authors instead stochastically sample points along the line and computes the minimum distances from any of the sample points to our original point. To make this method more robust for my purposes, I instead sample points uniformly along the curve.

Again there is no closed form solution to uniformly sample distances along a Bézier curve, so I use a linear approximation with a couple points.

### Optimization Results ###
<img src="./repo_assets/good_optimization_1.gif" alt="Good Optimization 1"/>
<p align = "center">Example of the curve optimizing to a target image in blue</p>

<img src="./repo_assets/good_optimization_2.gif" alt="Competitive Drawing Logo"/>
<p align = "center">The curve is capable of disentangling itself</p>

<img src="./repo_assets/local_minimum.gif" alt="Local Minimum"/>
<p align = "center">Like any gradient descent algorithm, there exist local minima</p>

<img src="./repo_assets/stuck_on_edge.gif" alt="Stuck on Edge"/>
<p align = "center">Local minima often exist at the edges. This is due to how the other key points push on the end points</p>

### Constraints and Curve Truncation ###
I enforce two constraints on my optimized curve.

The first is that the endpoints must stay on the canvas. This prevents situations such as the curve moving entirely off screen and thus producing no gradient to optimize.

The second is that the curve's arc length must not exceed a maximum length. As mentioned previously, the curve's length is estimated using a couple of points. I then compute the a [subdivision](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html) that satisfies my constraint using Casteljau's algorithm for computing Bézier curves.

<img src="./repo_assets/bezier_curve_subdivision.jpg" alt="Bezier Curve Subdivision" href="https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html"/>

Both of these constraints are non-differentiable, so I enforce them after each optimization step.

### Todo Board ###
https://kahl.notion.site/90a75c8cbbbe4eb1924d24e61818d2d2?v=9aeff125dd33457da8f046c1715c206a
