| class     |   recordings |   chance |   accuracy with the header field |   accuracy without it |   difference |
|:----------|-------------:|---------:|---------------------------------:|----------------------:|-------------:|
| DDoS-ICMP |           10 |      0.1 |                           0.8532 |                0.8702 |        0.017 |
| DDoS-SYN  |            5 |      0.2 |                           0.8784 |                0.8783 |      -0.0001 |
| DDoS-TCP  |            5 |      0.2 |                           0.8232 |                0.8856 |       0.0624 |
| DDoS-UDP  |           10 |      0.1 |                           0.6869 |                0.7157 |       0.0288 |
| DoS-ICMP  |            5 |      0.2 |                           0.7863 |                0.8339 |       0.0477 |
| DoS-SYN   |            5 |      0.2 |                           0.9177 |                0.9106 |      -0.0072 |
| DoS-TCP   |            5 |      0.2 |                           0.8291 |                0.8904 |       0.0613 |
| DoS-UDP   |            5 |      0.2 |                           0.8489 |                0.8186 |      -0.0303 |

Removing the header field raises the result on five of the eight classes and lowers it on three, one of them by a ten-thousandth.
