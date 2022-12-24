import os
import tqdm
import subprocess


def read_all_paths():
    all_paths = subprocess.Popen(
        "gsutil -m ls 'gs://quickdraw_dataset/full/numpy_bitmap'",
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.read()
    all_paths = all_paths.decode("utf-8")
    all_paths = all_paths.split("\n")
    all_paths = all_paths[:-1]

    return all_paths


if __name__ == "__main__":
    all_paths = read_all_paths()

    for category_path in tqdm.tqdm(all_paths):
        category_name = category_path.split("/")[-1].split(".")[0]
        category_name = category_name.replace(" ", "\ ")

        output_path = os.path.join("raw_data", f"{category_name}.npy")
        process = subprocess.Popen(f"gsutil -m cp '{category_path}' {output_path}", shell=True)
        #process.wait()
