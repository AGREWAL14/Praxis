| model                                                                    |     F1 | split                | averaging                |
|:-------------------------------------------------------------------------|-------:|:---------------------|:-------------------------|
| Logistic regression                                                      |  0.432 | by capture file      | not stated in the source |
| AdaBoost                                                                 |  0.141 | by capture file      | not stated in the source |
| Deep neural network                                                      |  0.522 | by capture file      | not stated in the source |
| Random forest                                                            |  0.551 | by capture file      | not stated in the source |
| Random forest, the published configuration, on a capture-disjoint split  |  0.868 | by recording session | macro                    |
| Random forest, the leaf setting chosen here, on a capture-disjoint split | 0.8418 | by recording session | macro                    |

Three things differ between the published figures and the runs here and none is controlled: the averaging method is not stated anywhere in the source, so the published figures may not be macro at all; the split is by capture file there and by recording session here; and the feature count is unsettled, with thirty-nine listed and forty-five shipped against the forty-four used here. This is a reference point showing where the work sits under a stricter protocol, not a controlled comparison between models.
