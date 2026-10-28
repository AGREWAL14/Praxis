# Captions

Descriptive captions for the tables and figures. Numbering is added when the
chapter is written, not here.

## model_comparison_nineteen_class

Nineteen-class results for four models. The rows are scored on different items and different held-out partitions, so they are four models' own scores rather than a paired comparison.

Note. The gap between weighted and macro F1 is the aggregate that a class-imbalanced task hides behind: accuracy and weighted F1 read high while the smallest classes are undetected.

## hypothesis_classes_by_model

The five classes the published single-record model scores below F1 0.50, under each of the four models. Four of the five reach the floor when records are read as sequences.

Note. Two of the four recovered classes rest on very few test sequences — 18 and 40 — so the direction is more interpretable than the magnitude.

## per_class_f1_single_record_against_sequence

Per-class F1 for all nineteen classes, the published single-record model against the model reading fifty-record windows, sorted by the difference between them.

Note. Nine classes gain and ten lose. The gains sum to +2.0566 and the losses to -2.0033, a net of +0.0533 across nineteen classes. 4 classes cross above F1 0.50 and 1 crosses below it. The five largest losses are all volumetric classes and account for 88% of the total negative movement.

## detection_floor_crossings_by_class

Per-class F1 under the published single-record model and under the model reading fifty-record windows, with the F1 0.50 detection floor marked. Classes marked with an asterisk rest on too few test sequences for the magnitude to be interpreted.

## class_balancing_interventions

Five treatments of class imbalance applied one at a time to the sequence model, on the five classes the published model fails, with the volumetric mean and macro F1 alongside. All rows are scored on the same test windows.

Note. No treatment raises the count of the five reaching the floor. One class does not respond to any of them, moving across a range narrower than a tenth of the floor, which points at how separable its traffic is rather than at how rare it is.

## cross_seed_macro_f1

Macro F1 across five training seeds for the sequence model at the full fifty-record window, with the one class that falls below the detection floor shown alongside.

Note. The class stays below the floor at every seed, so its fall is a property of window-based sequencing rather than of the seed the first run used. What causes it is not established here.

## paired_significance_tests

Paired tests between observation budgets and between the untreated model and each balancing treatment, with a correction applied within each family and again across all eleven.

Note. No test is computed against the published single-record model. It is scored on different units and a different held-out partition, so no item-level pairing between the two exists and no substitute was computed.

## class_pairs_below_auc_090

The class pairs no single feature separates at AUC 0.90, out of all 171 pairs.

Note. These are the pairs where the difficulty is in the traffic rather than in the class sizes, which is the reading the balancing results point to for the one class that does not respond to any treatment.

## saturation_budget_by_class

The smallest number of observed records at which each class comes within 0.02 of its own best score across the budget grid, with the score it reaches there.

Note. Low-rate median 25 records against volumetric 15. 7 of nineteen classes stop improving at the smallest budget, and 8 saturate at an F1 below 0.80, so saturation marks where a class stops improving rather than where it becomes reliably detectable. The grid takes only four values, so a median can land on one of them or midway between two, and a finer grid would move the ratio between the groups.

## records_to_reach_the_reliable_level

How many observed records each class needs to reach F1 0.80, with classes that do not reach it within fifty records reported as exceeding the grid rather than as fifty.

Note. 8 of nineteen classes do not reach the level within fifty records. Because three of five low-rate classes are among them the low-rate median is not determinate on this scale, which is why saturation rather than this measurement carries the group comparison. Two situations sit behind the same marker: some of these classes are at their best score at fifty records and others peaked earlier and fell.

## macro_f1_by_observation_budget

Overall results at each observation budget. Every budget is scored on the same test windows and uses the same model size, so the only thing varying is how many records of each window the model sees.

Note. The curve is read as flat from twenty-five records rather than declining: the spread across seeds at fifty records is wider than the gap between the two.

## f1_by_observation_budget

Per-class F1 across the four observation budgets, separated into the two groups, with each class's saturation point ringed and the reliable level marked; and overall macro F1 across the same budgets with the spread across five training seeds at the largest budget. Dashed lines and asterisks mark classes resting on too few test sequences to interpret.

## cost_of_excluding_the_timing_features

Macro F1 for both models with and without the four timing features, which are the features that identify the recording session. The sequence figure is the mean across five training seeds.

Note. The two rows are scored on different units. Removing the same four columns costs the forest an order of magnitude more than it costs the sequence model. The forest is used here only as an exactness check on the attribution method, which holds whatever the model scores, but a reader told it is the exact comparator should also be told what it scores.

## attack_pattern_and_category_by_class

Each attack class, the five features its explanations rank highest, the attack pattern the mapping resolves from the ten it reads, the threat category that pattern carries, and the category documented for that class in the benchmark paper.

Note. The feature column shows the highest five. The mapping reads ten, and all ten are in `feature_ranking_by_class` with their attribution weights. Every class reaches an attack pattern. Half reach the category documented for them. One class rests on a single mapping entry — the one linking the address-resolution protocol to identity spoofing — and one is the only route to its category and rests on forty test sequences, so neither should be read as evidence about the category.

## feature_ranking_by_class

The ten features the mapping reads for each attack class, in rank order, with the mean absolute attribution behind each and its share of the ten.

Note. The share column is what says how much any assignment rests on one feature. Where the ten are near-flat, no single feature carries the pattern the class resolves to. The ranking is recomputed from the attribution matrix rather than read from threat_mapping.json, which stores only the highest five; the recomputed five reproduce that string for all eighteen classes.

## attack_pattern_and_category_by_class_exact_attributions

The same mapping applied to the single-record forest, whose attributions are exact rather than approximate. Used as a check on the attribution method, not as a test of the mapping.

Note. This model agrees with the documented category less often than the approximate one, which rules out approximation error as the explanation for the disagreements.

## feature_ranking_by_class_exact_attributions

The same ranking for the forest, whose attributions are exact rather than approximate.

Note. Read against `feature_ranking_by_class`, this shows whether the two models rank the same features or reach the same pattern by different routes.

## attack_pattern_resolution_summary

How far the mapping reaches and how often it reaches the right place. Reaching no attack pattern and reaching the wrong category are different failures and are counted separately.

Note. The mapping declines 20 of 40 features, so a class whose highest-ranked features all landed among them would have resolved to nothing and counted against the result. That every class resolved is therefore a measurement rather than an artifact of a table built to cover everything. Twelve of the eighteen classes are denial of service, so a rule that read nothing and answered denial of service every time would match the majority baseline; the agreement figure has to be read against it.

## attribution_stability_by_class

Rank agreement between the highest-ranked features of models trained with different seeds, over all ten pairs of five seeds, per class.

Note. Mean across the nineteen classes 0.4560. No pass mark applies to this measurement; it is reported alongside the mapping so a reader can see how stable the explanations the mapping reads actually are.

## attribution_weight_by_class_and_feature

How much each of the forty remaining features contributes to each class's explanations, scaled within each class so the pattern rather than the magnitude is comparable across rows. The four timing features are absent because the model is trained without them.

## adverse_event_reports_by_device_category

Adverse event reports held in the public postmarket surveillance database for the device categories represented in the testbed, over the window searched.

Note. Four of the eight categories return nothing at all, which is a finding about how the database names devices rather than about those devices being safe. Of 829 reports across the eight categories, 1 mentions any of the fifteen attack-related terms searched, a share of 0.0012. That count is equally consistent with such events being rare and with the database having no category in which to record them, and it cannot distinguish the two. It is reported as adverse events in these device categories mentioning these terms, never as harm caused by an attack.

## recording_session_identification_by_feature_set

How well a model names which recording session a row came from, by feature set, with and without the one column in the timing group that turns out to be a header field rather than a measure of time. The last row holds the attack class fixed, so telling one attack from another is no help.

Note. Four timing features identify the session better than all forty-four together. The result with the attack class held fixed does not depend on the header field and rises slightly when it is removed, so the finding cannot be dismissed as an artifact of one badly chosen column. That column alone identifies the session a little above chance rather than at chance. Every figure here is a single run at one seed with no replicates.

## recording_session_identification_by_class

Identifying which recording a row came from with the attack class held fixed, for the eight classes recorded more than once, with and without the header field.

Note. Removing the header field raises the result on five of the eight classes and lowers it on three, one of them by a ten-thousandth.

## published_benchmark_anchor

The originating benchmark's own published figures for this dataset, and the same forest configuration evaluated under a split that holds recording sessions apart.

Note. Three things differ between the published figures and the runs here and none is controlled: the averaging method is not stated anywhere in the source, so the published figures may not be macro at all; the split is by capture file there and by recording session here; and the feature count is unsettled, with thirty-nine listed and forty-five shipped against the forty-four used here. This is a reference point showing where the work sits under a stricter protocol, not a controlled comparison between models.

