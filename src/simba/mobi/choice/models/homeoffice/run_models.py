import subprocess

MAIN_FILE = "main.py"

models = ["dcm", "rumboost"]
intensity_cutoffs = [20, 10, 0]

for model in models:
    for cutoff in intensity_cutoffs:
        print(f"Running main.py with model={model} and intensity_cutoff={cutoff}")
        subprocess.run(["python", MAIN_FILE, "--model", model, "--intensity_cutoff", str(cutoff)])