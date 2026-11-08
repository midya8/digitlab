# Browser inference, without a prediction service

Open `docs/index.html` after cloning. No install or internet is needed for the
page itself. It contains the unchanged selected linear and tanh-MLP parameters,
ten attributed original UCI inputs, manual forward passes, input gradients,
and clipped FGSM. No trained browser framework or autograd engine is used.

## A label is not a prediction input

An original example supplies its recorded label. Painting, keyboard editing or
clearing the input clears it. Ordinary inference needs no label. FGSM needs an
explicit loss label, so the attack button stays disabled until a label is
chosen. Editing never silently reuses the source example's label. The attack
result is separate from the original editable input. Model, label and radius
changes invalidate a displayed attack result.

Mouse painting, touch tapping/dragging and keyboard navigation are supported.
Tab enters the grid at one cell, arrows move focus, and Space/Enter paints.
The spatial 8x8 controls are small on phones; keyboard input is an alternative.
No formal accessibility-conformance claim is made.

## Independent numerical references

`examples/build_web_data.py` generates `docs/data.js` from the bundled selected
models and the first official test row per class. Original model SHA-256 values
and test-row indices are preserved. Ten first-per-class examples are a
repeatable demonstration, not a representative performance sample.

The same exporter creates 52 NumPy reference cases: each of two models on the
first 20 official rows, zero/one inputs and four seeded uniform inputs. Node
checks all class probabilities, log probabilities, loss, every input-gradient
coordinate and clipped FGSM pixels against NumPy with explicit numerical
tolerances (absolute 2e-12, or 5e-12 for gradients). Regeneration keeps model/sample
bytes and discrete inputs/labels/predictions exact but compares finite floating
reference values with those tolerances, not a cross-machine bitwise promise.
Roundoff-acceptance and material-corruption/NaN/infinity/discrete-change rejection
controls test this boundary. These are finite checks, not an all-input proof.
Zero-radius identity, class-order ties, invalid inputs and explicit attack-label
requirements are tested separately. The original 59 Python tests remain intact.

A real headless Chrome test checks model selection, clear/restore, label clearing,
FGSM bounds, keyboard movement/painting, actual dispatched touch dragging, and
horizontal layout at 320, 390 and 1280 pixels. Node 22 and Chrome are required
only for developer browser tests, not visitors. Temporary profiles are deleted
with awaited retries after Chrome exits.

```bash
OPENBLAS_NUM_THREADS=1 python examples/build_web_data.py
python -m unittest discover -s tests -v
node --test tests/test_web.cjs tests/test_browser.cjs
```

## Privacy, hosting and limits

Inference and input changes stay in the page. There is no prediction request,
user-input upload, analytics, storage, service worker, external font or CDN.
On a hosted deployment, loading its static HTML and JavaScript contacts the
host normally. GitHub Pages logs visitor IP addresses for security regardless
of sign-in: https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages .
Use the local files if static-host contact is unwanted.

Softmax scores are not calibrated confidence. Edited block counts are not a
camera image, production OCR, or an accuracy benchmark. The stronger 1NN clean
result remains visible but its classifier is not executed in the page. FGSM is
one bounded loss-increasing attack, not a proof of robustness, imperceptibility
or preservation of a human label. Exact rational affine proofs in BOX.md and
RADIUS.md remain separate and are not run by JavaScript. In particular, neither
those proofs nor finite NumPy reference checks certify every floating-point
boundary tie or the nonlinear MLP.

Data/adapted visualizations retain the attribution and CC BY 4.0 terms in
[docs/NOTICE.md](docs/NOTICE.md). Original code is MIT. No endorsement is implied.
