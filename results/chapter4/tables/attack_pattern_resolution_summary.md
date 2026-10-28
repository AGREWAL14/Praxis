| measure                                                    | value    |   as a proportion | compared against                                                        |
|:-----------------------------------------------------------|:---------|------------------:|:------------------------------------------------------------------------|
| Attack classes reaching an attack pattern                  | 18 of 18 |                 1 | a pass mark of 15 of 18                                                 |
| Features the mapping declines to map                       | 20 of 40 |               0.5 | a class whose highest-ranked features all fell here would reach nothing |
| Classes whose resolved category matches the documented one | 9 of 18  |               0.5 | a majority-class baseline of 0.667                                      |
| The same, for the model with exact attributions            | 6 of 18  |             0.333 | the same baseline                                                       |

The mapping declines 20 of 40 features, so a class whose highest-ranked features all landed among them would have resolved to nothing and counted against the result. That every class resolved is therefore a measurement rather than an artifact of a table built to cover everything. Twelve of the eighteen classes are denial of service, so a rule that read nothing and answered denial of service every time would match the majority baseline; the agreement figure has to be read against it.
