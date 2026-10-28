| model                                                              |   with all forty-four features |   with the four timing features removed |   fall | test unit   |
|:-------------------------------------------------------------------|-------------------------------:|----------------------------------------:|-------:|:------------|
| Random forest, single records                                      |                         0.8418 |                                   0.591 | 0.2508 | records     |
| Convolutional encoder with a recurrent layer, fifty-record windows |                         0.7138 |                                  0.6901 | 0.0237 | windows     |

The two rows are scored on different units. Removing the same four columns costs the forest an order of magnitude more than it costs the sequence model. The forest is used here only as an exactness check on the attribution method, which holds whatever the model scores, but a reader told it is the exact comparator should also be told what it scores.
