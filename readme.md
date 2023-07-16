To do

- Fix training and validation progress bars
- Add `requirements.txt`
- Add installation instructions
- Clean up plotting, move this to utils
- Fix randomness seed for validation (same noise for all validation epochs)
- Train two models with:
    - Two different lengthscale initialisations
    - Batch size 16 
    - Same noise as in previous implementation
    - Same context size as in previous implementation
- Add amortisation for `y_bound` and `w_noise`
- Train full set of models on EQ tasks