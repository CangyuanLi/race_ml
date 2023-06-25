# race_ml:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

This repository contains the code (and selected data) necessary to train the model used by
the **pyethnicity** package. The model uses L2 voter registration data from all 50 states in
combination as its source of names, zip codes, and self-reported race. This repository is still
in active development.

Currently, the best-performing model is a Dual-Bidirectional LSTM that uses first name,
last name, and the racial distribution of the person's ZCTA as inputs. It trains an LSTM
on the first name, and LSTM on the last name, concatenates the outputs, and applies
a final softmax activation. Additionally, first and last names are passed through
an embedding layer. It's hyperparameters are:

- Batch Size: 512
- Loss Function: NLLLoss
- Dropout: 0.2
- Epochs: 15
- Embedding Dimension: 512
- Hidden Size: 128
- Number of LSTM Layers: 4
- Learning Rate: 0.0001
- Optimizer: AdamW
- Weight Decay: 0.0001

Here is a comparison of the model with available packages:

**pyethnicity**

|    | race     |   accuracy |   precision |   recall |   f1_score |       fpr |   support |
|---:|:---------|-----------:|------------:|---------:|-----------:|----------:|----------:|
|  0 | asian    |    0.92882 |    0.842357 |  0.87996 |   0.860748 | 0.0548933 |         1 |
|  1 | black    |    0.88278 |    0.769988 |  0.75736 |   0.763622 | 0.0754133 |         1 |
|  2 | hispanic |    0.95436 |    0.911717 |  0.90508 |   0.908387 | 0.0292133 |         1 |
|  3 | white    |    0.8627  |    0.730226 |  0.71492 |   0.722492 | 0.08804   |         1 |

**rethnicity**

|    | race     |   accuracy |   precision |   recall |   f1_score |      fpr |   support |
|---:|:---------|-----------:|------------:|---------:|-----------:|---------:|----------:|
|  0 | asian    |    0.92773 |    0.849959 |  0.86332 |   0.856587 | 0.0508   |         1 |
|  1 | black    |    0.84771 |    0.680031 |  0.73816 |   0.707904 | 0.115773 |         1 |
|  2 | hispanic |    0.94701 |    0.930059 |  0.85212 |   0.889385 | 0.02136  |         1 |
|  3 | white    |    0.83561 |    0.674252 |  0.66252 |   0.668335 | 0.106693 |         1 |

**ethnicolr**

|    | race     |   accuracy |   precision |   recall |   f1_score |       fpr |   support |
|---:|:---------|-----------:|------------:|---------:|-----------:|----------:|----------:|
|  0 | asian    |    0.89371 |    0.991585 |  0.57976 |   0.731706 | 0.00164   |         1 |
|  1 | black    |    0.83101 |    0.833018 |  0.40528 |   0.545274 | 0.02708   |         1 |
|  2 | hispanic |    0.94822 |    0.927051 |  0.8606  |   0.89259  | 0.0225733 |         1 |
|  3 | white    |    0.72872 |    0.478725 |  0.95768 |   0.638351 | 0.3476    |         1 |
