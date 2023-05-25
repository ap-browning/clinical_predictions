# clinical_predictions
 
Repository associated with the preprint ["Predicting radiotherapy patient outcomes with real-time clinical data
 using mathematical modelling"](https://arxiv.org/abs/2201.02101).

- Each figure in the article can be reproduced by running the corresponding file in the `figures` subdirectory (for example, `figures/fig1.jl`).
- In the `analysis`, we provide code used to calibrate the model at the population level, code used to classify patients, code used to numerically solve the mathematical model, in addition to `csv` files containing samples from the second level prior that are used to reproduce results in the main document.
- In the `data` directory, we provide an example synthetic data set, in the same format and size as the clinical data set used in the main document.
    - To run the code with the synthetic data set provided, change references to `data/data_volumes.csv` and `data/data_dosetimes.csv` in `analysis/data.jl` to reflect the filenames of the synthetic data set.
    - Please contact [heiko.enderling@moffit.org](mailto:heiko.enderling@moffit.org) for queries relating to access to the actual clinical data set.