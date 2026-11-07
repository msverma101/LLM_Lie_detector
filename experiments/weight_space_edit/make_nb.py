"""Builds wsp.ipynb (Colab). Run: python make_nb.py"""
import json
BR = "research/wsp-weight-edit"
cells = [
 ("md", "# WSP arm: Claude edits the trained probe's weights directly\n\nθ_new = θ_old + Δθ_Claude (no retraining). Compared against gradient fine-tuning, refit, random edits and the unedited probe. Calibration and test are separate; test is unsealed once.\n\n1. Runtime → GPU (T4). 2. Secret `ANTHROPIC_API_KEY`. 3. Run all."),
 ("code", f"BRANCH = '{BR}'\n!rm -rf LLM_lie_detector && git clone -q -b $BRANCH https://github.com/msverma101/LLM_lie_detector.git\n%cd LLM_lie_detector/experiments/research_loop\n!git log --oneline -1\n!pip install -q -U transformers accelerate anthropic huggingface_hub"),
 ("md", "## 1. Activations (same cache as the research loop, ~13-20 min on a T4)"),
 ("code", "!python cache_activations.py --model Qwen/Qwen3.5-2B --n_per_family 200 --out acts"),
 ("md", "## 2. Train the probe once, save θ0, freeze partitions"),
 ("code", "%cd ../weight_space_edit\n!python wsp.py prepare --acts ../research_loop/acts --run runs_wsp/run1"),
 ("md", "## 3. Ordinary optimization of the same parameters (GD fine-tune + refit)"),
 ("code", "!python wsp.py gd --acts ../research_loop/acts --run runs_wsp/run1"),
 ("md", "## 4. WSP: Claude Opus 5.5 edits the weights (only dev_inspect + aggregate selection scores)"),
 ("code", "import os\nfrom google.colab import userdata\nos.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')\n!python wsp.py claude --acts ../research_loop/acts --run runs_wsp/run1"),
 ("md", "## 5. Random-edit control, then the one-time final evaluation"),
 ("code", "!python wsp.py random --acts ../research_loop/acts --run runs_wsp/run1\n!python wsp.py final --acts ../research_loop/acts --run runs_wsp/run1"),
 ("code", "!cd runs_wsp && zip -qr ../wsp_run1.zip run1 -x 'run1/candidates/*.npz' 'run1/*.npz'\nfrom google.colab import files\nfiles.download('wsp_run1.zip')"),
]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": "markdown" if t == "md" else "code", "metadata": {}, "source": s, **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells]}
json.dump(nb, open("wsp.ipynb", "w"), indent=1)
