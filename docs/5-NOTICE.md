# Browser demo provenance

No user input is sent to a server. All inference and FGSM computations are local
JavaScript. No analytics, font CDN, model API, service worker or external runtime
is used. Loading the page/assets still contacts its static host normally.

Models are the unchanged selected linear/MLP JSON parameters. The deterministic
exporter in examples/build_web_data.py records their original SHA-256 values.
Ten examples use the first official test row for each class, explicitly not a
representative sample. Grayscale digit adaptations and further edits/attacks
retain CC BY 4.0: Alpaydin & Kaynak (1998), UCI Optical Recognition of Handwritten
Digits, https://doi.org/10.24432/C50P49 . Source and license:
https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
https://creativecommons.org/licenses/by/4.0/

Original code is MIT. FGSM and manual softmax/tanh backpropagation are established
methods, not inventions. Exact rational proofs in the repository are NOT run by
this floating JavaScript demo. Sources and limitations remain in METHOD.md,
BOX.md and RADIUS.md at https://github.com/midya8/digitlab .
