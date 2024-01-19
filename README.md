# clinical_predictions
 
Repository associated with the paper ["Predicting radiotherapy patient outcomes with real-time clinical data
 using mathematical modelling"](https://link.springer.com/article/10.1007/s11538-023-01246-0).

- Each figure in the article can be reproduced by running the corresponding file in the `figures` subdirectory (for example, `figures/fig1.jl`).
- In the `analysis`, we provide code used to calibrate the model at the population level, code used to classify patients, code used to numerically solve the mathematical model, in addition to `csv` files containing samples from the second level prior that are used to reproduce results in the main document.
- In the `data` directory, we provide an example synthetic data set, in the same format and size as the clinical data set used in the main document.
