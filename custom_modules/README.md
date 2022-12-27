TODO:
0. Enforce distance
1. Implement random initialization strategy
2. Implement alias decay strategy
3. Implement score model
4. Evaluate which strategy is best

Stroke distance will be enforced by finding a good curve, then truncating inwards to outwards
    Pick the endpoint that's closer to the center
    When drawing, just truncate to length

    Alternatively, enforce during optimization by moving endpoint to a sampled point
