# Multifidelity paper studies
Code to perform case studies for [Jasa et al's 2022 Wind Energy Science paper on multifidelity optimization.](https://wes.copernicus.org/preprints/wes-2021-56/)

## Installation and case running instructions

1. Install the [WEIS framework](https://github.com/johnjasa/WEIS/tree/update_multifidelity) following the instructions listed in its README. This contains the multifidelity trust region method code.
2. Install [FLORIS v. 2.2.4](https://github.com/NREL/floris/releases/tag/v2.2.4) for case study 3.
3. Clone this repository to your local machine by running `git clone https://github.com/johnjasa/multifidelity_paper_studies`.
4. The code and models to run each of the case studies presented in the paper are within this repo. Each top-level folder contains three run scripts: `run_low_fidelity.py`, `run_high_fidelity.py`, and `run_multifidelity.py` to run the optimization cases presented in the paper.
