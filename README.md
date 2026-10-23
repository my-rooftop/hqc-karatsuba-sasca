# Running the Code for "From Multiplication to Recovery: A Single-trace Side-Channel Attack on HQC"

## 1. SASCA on Recursive Karatsuba
The code for Karatsuba SASCA in Section 4 of the paper is located in the `SASCA` folder.

### How to Run the Code:
The program implements SASCA on the Karatsuba multiplication in the reference implementation of HQC.
It uses the Belief Propagation (BP) algorithm over the LLR domain, including BP based on the Sum-Product Algorithm (LLR-SP) and BP based on the Normalized Min-Sum Algorithm (LLR-NMS).
Using SASCA, it recovers the secret keys of HQC-128.

#### Required Input Files:
To run `test.py`, the following input files are required:

- **Side-channel Prediction Files**:
  - `mask_a0.txt`: Probability distribution of side-channel leakage `y0` (Algorithm 2, line 7) using CNN.
  - `mask_a1.txt`: Probability distribution of side-channel leakage `y2` (Algorithm 2, line 7) using CNN.
  - `mask_ta.txt`: Probability distribution of side-channel leakage `y1` (Algorithm 2, line 7) using CNN.
  - `r_a0.txt`: Probability distribution of side-channel leakage `r0` (Algorithm 2, lines 9, 10) using CNN.
  - `r_a1.txt`: Probability distribution of side-channel leakage `r2` (Algorithm 2, lines 9, 10) using CNN.
  - `r_ta.txt`: Probability distribution of side-channel leakage `r1` (Algorithm 2, lines 9, 10) using CNN.


## 2. Single-trace Key Recovery under Valid Ciphertexts
The code for Single-trace Key Recovery is located in the `KeyRecovery` folder (Section 5 of the paper).

### How to Run the Code:
This program implements a turbo-like post-processing/key-recovery procedure and operates on bit-level vectors stored in text files.
All inputs represent binary vectors of length `PARAM_N` and are loaded as bit-packed `uint64_t` arrays (`VEC_N_SIZE_64` words).
Each bit corresponds to one coordinate of the vector; unused tail bits must be zero.

#### Required Input Files:
To run `turbo-solver.c`, the following input files are required:

- **Side-channel Prediction Files**:
  - `PRED_C_TXT`: Noisy prediction of the ciphertext vector `c`.
  - `PRED_Y_TXT`: Noisy prediction of the secret vector `y`.
  - `PRED_X_TXT`: Prediction or probability output for the secret vector `x` (used for initialization or comparison).

- **Ground-truth Reference Files** (used only for validation and debugging):
  - `C_TRUE_TXT`: True ciphertext vector `c`.
  - `Y_TRUE_TXT`: True secret vector `y`.
  - `X_TRUE_TXT`: True secret vector `x`.

- **Public and Known Parameter Files**:
  - `R1_TXT`: Sparse vector `r1`, used in the product `r1 * y`.
  - `R2_TXT`: Sparse vector `r2`, used in the product `r2 * x`.
  - `E_TXT`: Known fixed-weight error vector `e`.
  - `H_TXT`: Public vector `h`.
  - `S_TXT`: Vector defined as `s = x + h * y`.
  - `M_TXT`: Message-related vector `m`, used to construct `mG`.
