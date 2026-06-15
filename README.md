# MNIST Addition Experiment

This experiment compares three neuro-symbolic approaches on the MNIST-addition task.  
The model receives handwritten digit images and is trained using only the final sum label.

The three tested categories are:

| Category | Method | Framework |
|---|---|---|
| A | Loss / constraint | NeurASP |
| B | Proof / derivation | DeepProbLog |
| C | Structure / architecture | LNN |

All experiments use the same CNN, MNIST split, and random seed.

## How to run

First create the environments:

```bash
python3.10 -m venv venvs/venv_neurasp
venvs/venv_neurasp/bin/pip install torch torchvision clingo tqdm matplotlib

python3.10 -m venv venvs/venv_deepproblog
venvs/venv_deepproblog/bin/pip install torch torchvision pysdd problog
venvs/venv_deepproblog/bin/pip install -e repos/deepproblog

python3.11 -m venv venvs/venv_lnn
venvs/venv_lnn/bin/pip install torch torchvision "git+https://github.com/IBM/LNN.git"
```

DeepProbLog also needs SWI-Prolog:

```bash
brew install swi-prolog
```

To run everything:

```bash
zsh run_local.sh
```

To run one setup manually:

```bash
venvs/venv_neurasp/bin/python shared/run_neurasp.py --n 2
venvs/venv_deepproblog/bin/python shared/run_deepproblog.py --method exact --n 1
venvs/venv_lnn/bin/python shared/run_lnn.py
```

To rebuild the tables and plots only:

```bash
venvs/venv_neurasp/bin/python shared/make_figures.py
```

## Output plots

The script creates four plots in the `plots/` folder:

- `accuracy_by_setup.png`  
  Shows sum accuracy and digit accuracy for the completed runs.

- `learning_curves.png`  
  Shows how sum accuracy changes during training.

- `compute_cost_vs_digits.png`  
  Shows the training time per step as the number of digits increases.

- `capability_matrix.png`  
  Shows which category works, runs separately, or fails for each digit setting.

## Notes

DeepProbLog exact does not finish for 3-digit addition because the inference circuit becomes too large.

LNN is run separately because it cannot train the CNN end-to-end from only sum labels. Its result is therefore reported separately and should not be directly compared with NeurASP and DeepProbLog.

The experiment uses one seed and runs on CPU, so the timing results are only indicative.