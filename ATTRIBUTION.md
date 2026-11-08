# A score difference is not a causal explanation

This extension evaluates established straight-line Integrated Gradients of a
chosen **class logit**, not class probability, correctness or a human concept.
The baseline is explicit and its nonzero logit is retained. All-zero block
counts and the frozen fitting-partition mean are comparisons, not claims of
neutrality, absence or realism. No test data enters the fitting mean.

## Algebra for the existing architectures

For a linear class logit F_k(x)=w_k^T x+b_k, each coordinate attribution is
(x_i-r_i) w_ik. The bias cancels from the baseline-to-input change.

For F_k(x)=sum_j v_jk tanh(w_j^T x+c_j)+d_k, put u_j=w_j^T r+c_j and
z_j=w_j^T x+c_j. The straight-line path has preactivation u_j+alpha(z_j-u_j).
Its integrated derivative is

    s_j = [tanh(z_j)-tanh(u_j)] / (z_j-u_j),
    s_j = sech(u_j)^2 when z_j=u_j,
    IG_i = (x_i-r_i) sum_j w_ij v_jk s_j.

This follows by integrating the derivative of tanh. It is an algebraic
specialization of an established method, not a new attribution method or an
exact floating-point certificate. For small preactivation changes, a symmetric
midpoint identity avoids subtracting nearly equal tanh values; sech-squared is
computed using exp(-2*abs(midpoint)) to retain small saturated derivatives.

The mathematical sum equals F_k(x)-F_k(r). The implementation reports the
floating residual instead of assuming it is zero. Small residual alone does
not verify individual feature assignments, nor does it prove usefulness or
faithfulness of an explanation. Independent Gauss-Legendre integration checks
individual coordinates, not just their sum.

## Controls and limits

Tests cover affine displacement/bias cancellation, zero-displacement and
unchanged-coordinate zeros, independent quadrature, small denominators and
saturation, function-preserving hidden-unit permutation/sign changes, explicit
baseline changes, all official inputs and strict input validation. These are
finite numerical checks. Extreme arbitrary parameters can have cancellation
and loss of accuracy; the implementation does not certify all binary64
inputs or all model weights. Baseline choice changes the question being asked.
A contribution sign is not the effect of removing a feature. Off-distribution
intermediate path points may not look like real handwriting.

Model-randomization checks, if reported, are descriptive controls, not a proof
that a trained model is trustworthy or that its feature attribution is causal.
The stronger same-fit 1NN clean baseline remains unchanged and is not given a
fake differentiable attribution.

## Sources

- Sundararajan, Taly & Yan (ICML 2017), original Integrated Gradients,
  straight-line path and completeness:
  https://proceedings.mlr.press/v70/sundararajan17a.html
  https://arxiv.org/html/1703.01365v2
- Captum API, independent Gauss-Legendre reference approach and residual:
  https://captum.ai/api/integrated_gradients.html
- TensorFlow tutorial, interpolation, gradient and integral computation:
  https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
- Adebayo et al. (NeurIPS 2018), why visual appeal and parameter insensitivity
  need checks: https://proceedings.neurips.cc/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf
- Independent textbook discussion of baseline-relative interpretation and
  evaluation limits: https://christophm.github.io/interpretable-ml-book/pixel-attribution.html
- Distill source repository, baseline choice and missingness assumptions:
  https://github.com/distillpub/post--attribution-baselines

No Captum, PyTorch, TensorFlow, autograd service or paid API is used.

## Frozen real-data audit

`attribution-report.json` keeps both baseline vectors, the fitting-index digest,
original dataset/model hashes, all official-row finite/residual summaries and
40 first-per-label reference vectors. Targets are each trained model's clean
prediction, not an assertion of correctness. The fitting mean loads only the
original training file and its fixed 3,062 fitting rows. The selected digit maps
in the figure use the first official row with recorded label 6, not a favorable
handpicked example. Both baselines share a symmetric scale within each model;
the interactive page instead states its own per-map scale.

All 7,188 row/model/baseline combinations have 460,032 finite attribution
coordinates. The saved maximum completeness residual is
1.0658141036401503e-14; maximum Gauss-Legendre coordinate difference on the 40
fixed demonstration vectors is 3.552713678800501e-15. Fitting-mean versus zero
maps differ in all 1,797 rows of both models at L1 threshold 1e-10. This is a
descriptive baseline comparison, not causal mechanism isolation or a score of
explanation quality. Saved-report reproduction uses finite absolute tolerance
5e-12 for floats and exact keys, shapes, discrete fields and hashes. Its guard
is tested against hidden material changes, nonfinite values and missing keys.

The browser independently matches 106 NumPy attribution vectors with absolute
tolerance 5e-12, including both models/references and zero-displacement cases.
Independent small-model finite-difference path integration and function-preserving
hidden permutations/sign changes add checks beyond completeness. This does not
prove all possible floating executions or baseline appropriateness.
