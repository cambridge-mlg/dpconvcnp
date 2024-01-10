# Differentially private conditional convolutional neural processes


Things to do:

- Recover lower bound performance with DPCovnCNP (data noise only)
    - ~~Modify UNet architecture~~
    - Long training run with:
        1. ~~Less frequent evaluation (i.e. more tasks per epoch)~~
        2. ~~Add noise to data channel only, fix `y_bound`, `w_noise`~~
        3. ~~Slightly larger convolutional model~~
        4. ~~Train one model per epsilon-lengthscale pair~~
    - ~~Repeat above experiment with amortisation over lengthscales~~
    - ~~Repeat above experiment with amortisation over epsilon~~
    - ~~Repeat above experiment with amortisation over lengthscales and epsilon~~
- Remove layernorm
- Create "gap plot" with three models:
    1. Data noise only
    2. Data noise and clipping
    3. Data noise, clipping, and density noise
- Add non-Gaussian datasets:
    1. Sawtooth
    2. Mixture of tasks