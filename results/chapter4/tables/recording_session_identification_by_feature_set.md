| feature set                               |   features |   chance | accuracy with the header field   | accuracy without it   | difference   |
|:------------------------------------------|-----------:|---------:|:---------------------------------|:----------------------|:-------------|
| All forty-four features                   |         44 |     0.02 | 0.801                            | 0.7495                | -0.0515      |
| Timing features only                      |          4 |     0.02 | 0.9301                           | 0.8935                | -0.0366      |
| Protocol features only                    |         28 |     0.02 | 0.1802                           |                       |              |
| Statistical features only                 |         12 |     0.02 | 0.1198                           |                       |              |
| The five features named by published work |          5 |     0.02 | 0.9285                           |                       |              |
| The header field alone                    |          1 |     0.02 |                                  | 0.0299                |              |
| All features, attack class held fixed     |         43 |    0.175 | 0.828                            | 0.8504                | 0.0225       |

Four timing features identify the session better than all forty-four together. The result with the attack class held fixed does not depend on the header field and rises slightly when it is removed, so the finding cannot be dismissed as an artifact of one badly chosen column. That column alone identifies the session a little above chance rather than at chance. Every figure here is a single run at one seed with no replicates.
