| class                   |   published |   sequence |   difference |   test sequences | group      | thin   |
|:------------------------|------------:|-----------:|-------------:|-----------------:|:-----------|:-------|
| DoS-ICMP                |       0.996 |     0.3178 |      -0.6782 |             3936 | volumetric |        |
| DoS-TCP                 |       0.998 |     0.5161 |      -0.4818 |             3282 | volumetric |        |
| DDoS-ICMP               |      0.9991 |     0.7721 |      -0.2271 |             7826 | volumetric |        |
| DoS-SYN                 |      0.9979 |     0.8023 |      -0.1955 |             3942 | volumetric |        |
| DDoS-TCP                |      0.9972 |     0.8137 |      -0.1835 |             7302 | volumetric |        |
| MQTT-DDoS-Publish_Flood |      0.1858 |     0.0796 |      -0.1061 |              215 |            |        |
| DDoS-SYN                |      0.9959 |     0.8926 |      -0.1033 |             6894 | volumetric |        |
| Recon-Ping_Sweep        |      0.0107 |          0 |      -0.0107 |                4 |            | yes    |
| DDoS-UDP                |      0.9985 |     0.9881 |      -0.0104 |             6255 | volumetric |        |
| DoS-UDP                 |      0.9946 |      0.988 |      -0.0066 |             5501 | volumetric |        |
| MQTT-DDoS-Connect_Flood |      0.9971 |     0.9996 |       0.0026 |             1288 |            |        |
| MQTT-DoS-Connect_Flood  |      0.9678 |     0.9741 |       0.0063 |               94 |            |        |
| Recon-Port_Scan         |      0.9006 |     0.9301 |       0.0295 |              638 | low-rate   |        |
| MQTT-DoS-Publish_Flood  |      0.6937 |     0.7551 |       0.0614 |              316 |            |        |
| Benign                  |      0.9094 |     0.9808 |       0.0714 |             1381 |            |        |
| MQTT-Malformed_Data     |      0.4883 |     0.8421 |       0.3538 |               40 | low-rate   | yes    |
| Spoofing                |      0.3438 |     0.7653 |       0.4215 |              104 | low-rate   |        |
| Recon-VulScan           |           0 |     0.5385 |       0.5385 |               18 | low-rate   | yes    |
| Recon-OS_Scan           |      0.0343 |     0.6061 |       0.5718 |              123 | low-rate   |        |

Nine classes gain and ten lose. The gains sum to +2.0566 and the losses to -2.0033, a net of +0.0533 across nineteen classes. 4 classes cross above F1 0.50 and 1 crosses below it. The five largest losses are all volumetric classes and account for 88% of the total negative movement.
