# Stanford Center for AI Safety Annual Meeting 2024

### Technical AI Governance

* Need technical tools and processes to enforce AI governance regulations.
* How do we measure training compute used?
* How can the thoroughness of model evaluations be measured?
* How can the downstream impacts of a model be predicted?
* Benchmarks:
  * Not all benchmarks are the same quality when it comes to "robustness".
  * Need to follow some "best practices" when it comes to defining and using benchmarks, e.g. [BetterBench](https://betterbench.stanford.edu/).

### Algorithms for Validation

* Validation processes should be performed in a controlled, safe enviornment, such as in a simulation, instead of in the real world.
* A "validation algorithm" checks a "system" against a "specification" that outlines what we want that system to do / not do.
* The validation algorithm should output information such as failure analysis, formal guarantees, explanations, and runtime assurances.
* The "swiss cheese model" works well at combining multiple methods within a validation algorithm to overcome any limitations of a single method.
* Types of validation algorithm:
  * Sampling-based methods (falsification and failure distribution/probability).
  * Formal methods (reachability).
  * Runtime monitoring.
  * Explainability.
