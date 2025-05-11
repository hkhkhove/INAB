import subprocess
import os
import time


def run(pdb_files):
    for i, pdb_file in enumerate(pdb_files):
        print(f"Processing {i + 1}/{len(pdb_files)}: {pdb_file}")
        prot_name, _ = os.path.splitext(os.path.basename(pdb_file))
        category = prot_category[prot_name]
        predict_start = time.time()
        subprocess.run(
            [
                "python",
                "predict.py",
                "--pdb_path",
                pdb_file,
                "--model_path",
                f"./model_parameters/INAB_{category}.pth",
            ]
        )
        predict_end = time.time()
        with open(pdb_file.replace(".pdb", "_time.txt"), "a") as f:
            f.write(f"total time: {predict_end - predict_start:.2f}s\n")


if __name__ == "__main__":
    prot_category = {}
    with open("./dataset/efficiency_test/prot_category.txt", "r") as f:
        for line in f:
            prot, category = line.strip().split()
            if category == "NONE":
                category = "DNA"
            prot_category[prot] = category

    pdb_files = [
        os.path.join("./dataset/efficiency_test/features", e)
        for e in os.listdir("./dataset/efficiency_test/features")
        if e.endswith(".pdb")
    ]
    pdb_files = [
        pdb_file
        for pdb_file in pdb_files
        if not os.path.isfile(pdb_file.replace(".pdb", "_output.txt"))
    ]

    run(pdb_files)
