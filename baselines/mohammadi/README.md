# baselines/mohammadi

The convolutional network described in Mohammadi, Amini, Karimi, Bagheri and Kaur
(2024), *Enhancing IoT Security: A Deep Learning-Based Approach* (arXiv:2410.23306),
evaluated on CICIoMT2024.

This directory is the reproduction. It is written once and then left alone.

Nothing here is tuned, corrected or improved, including the parts of it that are
plainly suboptimal on this dataset: there is no class weighting, no early
stopping, no validation split during training, and the input tensor is
`(samples, features, 1)`, so the convolution slides across the feature axis and
each sample is one record rather than a stretch of traffic. Those are properties
of the published model and they are the reason it is here. A baseline that has
been quietly repaired is not a baseline.

The scaler is fitted inside this directory, on the training side of the split it
is given, and the CSVs are read here as well. The project's own preprocessing in
`src/` is not used, because a reproduction whose value is that it was produced the
way the original was cannot have this project's preprocessing inside it.

| File | Holds |
|---|---|
| `cnn.py` | the architecture, the compile settings and the fit settings |
| `data.py` | reading the capture files, the scaler, and the label encoding |

Two things are supplied by the caller rather than fixed here: which columns to
read, and which files are on the training side. Both are properties of the data
and the split, not of the model.

The one accommodation made to this environment is that `fit` accepts a list of
Keras callbacks. Callbacks are passed a copy of the training state after each
epoch and cannot alter it; the checkpoint the notebook attaches writes the model
to Drive so a dropped Colab session does not cost the whole run. No callback
that changes training is passed, and there is no early stopping.
