**Final project proposal: When do quantum circuits keep federated training data private?**

In federated learning, clients share gradients instead of data, and on classical neural networks
those gradients can be inverted to reconstruct the training inputs. Kumar et al. (2023) argue that
variational quantum circuits with expressive encodings resist this attack, while Papadopoulos et al.
(2025) reconstruct inputs once the circuit is overparameterized. I will measure reconstruction error
across encoding depth, trainable layers, batch size, and simulated IBM Heron device noise to locate
where that privacy holds and where it breaks. The deliverable is a LaTeX report and an in-class
presentation of the resulting privacy map.

References: arXiv 2309.13002; arXiv 2504.12806.
