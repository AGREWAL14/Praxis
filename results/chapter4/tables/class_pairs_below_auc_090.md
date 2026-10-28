| first class             | second class        | same attack family   | best single feature   |   best AUC | next best feature   |   next best AUC |
|:------------------------|:--------------------|:---------------------|:----------------------|-----------:|:--------------------|----------------:|
| Recon-OS_Scan           | Recon-Port_Scan     | yes                  | Rate                  |      0.685 | Srate               |           0.685 |
| Recon-Port_Scan         | Recon-VulScan       | yes                  | Duration              |     0.7368 | Radius              |          0.7326 |
| Recon-OS_Scan           | Recon-VulScan       | yes                  | Radius                |     0.7442 | Std                 |          0.7441 |
| MQTT-Malformed_Data     | Spoofing            |                      | TCP                   |     0.7847 | UDP                 |          0.7736 |
| Benign                  | Spoofing            |                      | Rate                  |      0.801 | Srate               |           0.801 |
| MQTT-DoS-Publish_Flood  | MQTT-Malformed_Data | yes                  | psh_flag_number       |     0.8514 | HTTPS               |          0.8207 |
| Benign                  | MQTT-Malformed_Data |                      | Rate                  |     0.8664 | Srate               |          0.8664 |
| Recon-VulScan           | Spoofing            |                      | Header_Length         |     0.8671 | Min                 |          0.8065 |
| MQTT-DoS-Publish_Flood  | Spoofing            |                      | psh_flag_number       |     0.8733 | ack_flag_number     |          0.8055 |
| MQTT-DDoS-Publish_Flood | MQTT-Malformed_Data | yes                  | fin_count             |     0.8753 | rst_count           |          0.8246 |
| MQTT-DDoS-Publish_Flood | Spoofing            |                      | fin_count             |      0.893 | rst_flag_number     |          0.8754 |

These are the pairs where the difficulty is in the traffic rather than in the class sizes, which is the reading the balancing results point to for the one class that does not respond to any treatment.
