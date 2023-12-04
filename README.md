# Differentially private conditional convolutional neural processes


Things to do:

- Recover lower bound performance with DPCovnCNP (data noise only)
    - Modify UNet architecture
    - Long training run with:
        1. Add noise to data channel only, fix `y_bound`, `w_noise`
        2. Slightly larger convolutional model
        3. Train one model per epsilon-lengthscale pair
    - Repeat above experiment with amortisation over lengthscales
- Create "gap plot" with three models:
    1. Data noise only
    2. Data noise and clipping
    3. Data noise, clipping, and density noise
- Add non-Gaussian datasets:
    1. Sawtooth
    2. Mixture of tasks