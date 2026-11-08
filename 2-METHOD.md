# Protocol, provenance and limits

## Scope and data boundary

DigitLab is an original NumPy implementation of small multiclass models and
input attacks on a historical public dataset. It is deliberately different
from a synthetic exact linear-margin certificate: its trained MLP is nonlinear,
its gradients are floating-point, and its finite attacks do not certify a ball.
There is no convolution, image-file OCR pipeline, language model or new algorithm.

Data are the unmodified UCI `optdigits.tra` and `optdigits.tes` numeric files.
The [manifest](digitlab/data/manifest.json) records original bytes, row counts,
file SHA-256 values and the downloaded archive SHA-256. The download was taken
from the live UCI dataset page, whose license explicitly says CC BY 4.0. The
separate [data notice](digitlab/data/LICENSE.md) preserves attribution and links
to the license; the project's MIT license applies to original code, not data.
Rendered digits are grayscale adaptations, and attack images are additionally
perturbed adaptations. The authors and UCI do not endorse these results.

Official split:

- 3,823 original training rows from 30 writers, according to UCI.
- 1,797 original test rows from a different 13 writers, according to UCI.
- Each row: 64 integer block counts in [0,16], followed by a digit label 0..9.
- Original 32-by-32 bitmaps were converted to 8-by-8 block counts by the dataset
  authors, not this project. This is not raw 28-by-28 MNIST.

Individual writer IDs are not present. We preserve the official separation but
cannot independently verify identities or make a writer-disjoint validation
split inside training. No numeric row is duplicated across fit/validation/test
by exact feature SHA-256. Similar handwriting, near duplicates and family/style
correlation are not ruled out by that check.

The scikit-learn `digits` dataset is a copy of the **official test file**, not
the complete 5,620-example benchmark. We therefore do not randomly split that
1,797-row pool and describe it as the original writer-independent evaluation.

## Fixed preprocessing and selection

Divide every feature by 16, a constant from metadata. Do not estimate a scale or
center using any partition. For each class in official training, hash-sort the
64 original feature bytes and hold out the first `floor(class_count/5)` rows.
The result is 3,062 fitting rows and 761 validation rows. Ties use original row
index, and the final index lists are sorted and saved in `protocol.json`.

The protocol was fixed before loading test examples:

| Setting | Value |
|---|---|
| Initialization seed | 23 |
| Shuffle seed | 41, reset per model |
| Models | Linear softmax; one tanh hidden layer of width 32 |
| Classes / inputs | 10 / 64 |
| Epochs / batch size | 80 / 128 |
| Learning rate / momentum | 0.08 / 0.9 |
| Weight L2 coefficient | 0.0001, biases excluded |
| Candidate checkpoints | 10, 20, 40, 80 |
| Selection | Minimum validation NLL, earliest checkpoint on ties |

Both checkpoints selected epoch 80. In the MLP, epoch 40 gets 743 validation
examples correct, while epoch 80 gets 741. It is still epoch 80 that wins on
predeclared validation NLL (about 0.08427 versus 0.09256). We do not silently
switch selection criteria after seeing accuracy. Full candidate records,
selected model hashes and protocol hash are saved in `selection.json`.

After those choices, `evaluation-plan.json` fixes the attack grid and settings,
then test examples are loaded. Label-aware attack construction uses test labels
only to define the attack loss, not to train or select weights or attack radii.
Identical re-evaluations and refits are used to verify reproducibility. No new
setting or best seed is selected from test performance.

## Models and manual derivatives

Linear logits are `z = x W + b`. The MLP uses `h = tanh(x W1 + b1)`,
`z = h W2 + b2`. Hidden weights are initialized with seeded zero-mean Gaussians,
scale `sqrt(1/fan_in)`; biases are zero. Linear weights start at zero.

Use stable log-softmax: subtract the row maximum before exponentiation and
compute log probabilities directly, so an underflowed probability does not
turn finite log probabilities into an accidental `log(0)` error.

The minibatch objective is mean true-label cross entropy plus
`lambda/2 * sum(weight^2)`. The logit derivative is `(softmax(z)-one_hot(y))/n`.
Propagate through tanh with `1-h^2`, producing parameter gradients and the input
gradient of the *mean* objective. The positive batch-mean scale does not change
the sign used by the attacks. L2 contributes `lambda*W` to weights, not biases
or input derivatives. Momentum uses `v = 0.9*v - 0.08*gradient`, `p = p + v`.
There is no Nesterov, Adam, framework autograd or gradient clipping.

201 independent centered finite-difference probes check every parameter and
input coordinate for small linear/tanh models across three seeds, including
L2. These are tolerance-based checks, not exact arithmetic proofs for every
input. Separate tests check descent, extreme logits, L2/input-gradient separation
and no mutation of caller inputs or global random state.

1NN is a strong independent clean baseline. It stores the same 3,062 fitting
rows; exact squared Euclidean distances are computed in batches. Lowest fitting
index wins a distance tie. Direct per-query squared-difference references and
an explicit tie fixture check this implementation. UCI's source notes report
about 98% for 1NN using the original benchmark setup. Our 1,760/1,797 result uses
only the fitting subset and is not a claim to reproduce the original author's
all-training-row result exactly. There is no probability/NLL assigned to hard
1NN labels.

## Attack construction and denominators

For normalized input `x` and fixed true label `y`, FGSM computes
`clip(x + epsilon*sign(grad_x CE), 0, 1)`. Projection also explicitly enforces
`[x-epsilon, x+epsilon]`. Epsilon is in normalized units; multiply by 16 for
original block-count units. Continuous perturbed values need not correspond to
valid integer-valued image-block counts or actual handwriting.

Fixed evaluation grid: epsilon 0, 0.05, 0.1, 0.2 for FGSM and random sign; epsilon
0.1 and 0.2 for PGD. Random sign uses one seeded draw (903), same pattern across
radii and models, with identical projection. It is a descriptive control only,
not a distributional estimate, causal isolation or confidence interval.

PGD uses 20 steps of size epsilon/4, three random starts with seed 811, gradient
sign ascent, and projection after each step. The candidate pool includes clean,
FGSM, every random start and every iterate. Per example, any misclassification
beats any correct candidate, then larger true-label CE breaks ties in success
status. This is an explicit attack-goal retention rule, not an assertion that
cross entropy and 0/1 error always order candidates the same way. By construction
the pool cannot lose an already found FGSM/clean failure. Still, only a finite
candidate set is tested: attack failure is not a robustness certificate or an
optimal-adversary result.

Every report row gives full-test accuracy with denominator 1,797, plus attack
success on **initially correct** rows with a separate model-specific denominator
(1,734 for MLP, 1,698 for linear). All predictions, confusion matrices, actual
maximum L-infinity distance and pixel extrema are retained. Constraints are
checked up to an explicit 1e-12 floating tolerance. Zero-radius identity and
pool retention are independently tested.

Attack settings were not selected to match a target success rate. At epsilon
0.1 the MLP has 898 correct under FGSM, 768 under the PGD pool and 1,708 under
random sign. At epsilon 0.2 the PGD pool has 3 correct. These are empirical
results, not a guarantee about true labels under perturbation. No human
perceptual, real-world accessibility, privacy or security guarantee is inferred.

Transfer to 1NN evaluates the MLP's PGD images on the stored-example classifier.
It is not optimized against 1NN, so its better transfer accuracy must not be
reported as higher adaptive robustness. The illustration selects the lowest
three test indices among initially correct successes and among initially
correct survivors at MLP PGD epsilon 0.1. The selection rule is public and the
images are not presented as a representative sample.

## Packaging and reproduction

Seven Python modules, official numeric data, selected JSON models, protocol,
selection and full evaluation report are bundled. No pickle is read. Training
and evaluation outputs require an explicit directory. Default `demo` is a
read-only saved-report view, not a hidden retraining or external API call.

Thirty tests include the complete refit and complete evaluation. Test-loader
use is prohibited during the refit by a patched loader assertion. Refit weights
must match with `rtol=1e-7, atol=1e-8`; saved-model re-evaluation must reproduce
all predictions/counts and NLL to eight decimal places. Saved figure generation
is byte-exact from the frozen report. Cross-platform floating training is not
claimed bitwise identical. NumPy is pinned and CI limits BLAS threads for
repeatability, not as a portable performance promise.

The 62 displayed value cells and all 768 shown pixel values bind to source
report data. Every prediction, confusion table and selection record also lives
in compressed SVG metadata. Hidden-record changes change the figure. Source
numeric data retain their original file hashes; serialized model/protocol
hashes detect a changed selection artifact, not a maliciously rewritten codebase.

## Sources and attribution

All sources were retrieved November 8, 2026. Source models, tutorial code and
benchmark numbers are not copied into this implementation or claimed as our own.

1. [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), Alpaydin and Kaynak, 1998. Primary dataset provenance, original split, features and explicit CC BY 4.0 terms.
2. [Original optdigits.names](http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.names), UCI, July 1998. Original training/testing counts, class counts, author setup and 1NN reference. This is background, not a reproduction claim for our fitting subset.
3. Goodfellow, Shlens and Szegedy, [Explaining and Harnessing Adversarial Examples](https://arxiv.org/html/1412.6572v2), 2015 version. Primary FGSM paper. Its MNIST/ImageNet measurements are not these UCI measurements.
4. Madry et al., [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083), 2017 preprint, revised 2019. Primary projected-gradient/random-start context. We do not implement adversarial training or reproduce its benchmark networks.
5. PyTorch, [Adversarial Example Generation](https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html). Official explanation of label-aware input gradients, clipping and initially-correct attack denominators. This project does not use PyTorch or its pretrained model.
6. Stanford CS231n, [Neural Networks Part 2](https://cs231n.github.io/neural-networks-2/) and [Part 3](https://cs231n.github.io/neural-networks-3/). Teaching notes on initialization, regularization, gradient checks, SGD/momentum and validation. No teaching implementation is copied.
7. scikit-learn, [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html). Official classifier definition and limitations. We implement exact 1NN ourselves, not a scikit-learn estimator.
8. scikit-learn, [digits dataset source description](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/descr/digits.rst). Official clarification that the 1,797 bundled examples copy UCI's test set. Used to avoid pooling the wrong source; no scikit-learn dependency.
