import subprocess
import time
MAIN_FILE = "src/simba/mobi/choice/models/homeoffice/main.py"

models = ["dcm", "rumboost"]
intensity_cutoffs = [20, 10, 0]
start_time = time.time()
for model in models:
    for cutoff in intensity_cutoffs:
        print(f"Running main.py with model={model} and intensity_cutoff={cutoff}")
        subprocess.run(["python", MAIN_FILE, "--model", model, "--intensity_cutoff", str(cutoff)])

print(f"--- {time.time() - start_time} seconds ---")