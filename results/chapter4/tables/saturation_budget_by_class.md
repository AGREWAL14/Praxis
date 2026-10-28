| class                   | group      |   saturates at |   F1 there |   best F1 |   best at | thin   | below the reliable level   |
|:------------------------|:-----------|---------------:|-----------:|----------:|----------:|:-------|:---------------------------|
| MQTT-DDoS-Connect_Flood |            |              5 |     0.9973 |    0.9996 |        50 |        |                            |
| MQTT-DoS-Connect_Flood  |            |              5 |          1 |         1 |         5 |        |                            |
| MQTT-DoS-Publish_Flood  |            |              5 |     0.7488 |    0.7551 |        10 |        | yes                        |
| Benign                  |            |             10 |     0.9755 |    0.9808 |        50 |        |                            |
| MQTT-DDoS-Publish_Flood |            |             10 |     0.0889 |    0.0889 |        10 |        | yes                        |
| Recon-Ping_Sweep        |            |             25 |     0.8571 |    0.8571 |        25 | yes    |                            |
| Spoofing                | low-rate   |             10 |     0.7941 |    0.7941 |        10 |        | yes                        |
| MQTT-Malformed_Data     | low-rate   |             25 |     0.9067 |    0.9067 |        25 | yes    |                            |
| Recon-OS_Scan           | low-rate   |             25 |     0.6105 |    0.6105 |        25 |        | yes                        |
| Recon-Port_Scan         | low-rate   |             50 |     0.9301 |    0.9301 |        50 |        |                            |
| Recon-VulScan           | low-rate   |             50 |     0.5385 |    0.5385 |        50 | yes    | yes                        |
| DDoS-ICMP               | volumetric |              5 |     0.7998 |    0.7998 |         5 |        | yes                        |
| DDoS-SYN                | volumetric |              5 |     0.9449 |    0.9449 |         5 |        |                            |
| DDoS-TCP                | volumetric |              5 |     0.8132 |    0.8172 |        25 |        |                            |
| DoS-SYN                 | volumetric |              5 |     0.9073 |    0.9073 |         5 |        |                            |
| DDoS-UDP                | volumetric |             25 |     0.9815 |    0.9881 |        50 |        |                            |
| DoS-UDP                 | volumetric |             25 |     0.9805 |     0.988 |        50 |        |                            |
| DoS-ICMP                | volumetric |             50 |     0.3178 |    0.3178 |        50 |        | yes                        |
| DoS-TCP                 | volumetric |             50 |     0.5161 |    0.5161 |        50 |        | yes                        |

Low-rate median 25 records against volumetric 15. 7 of nineteen classes stop improving at the smallest budget, and 8 saturate at an F1 below 0.80, so saturation marks where a class stops improving rather than where it becomes reliably detectable. The grid takes only four values, so a median can land on one of them or midway between two, and a finer grid would move the ratio between the groups.
