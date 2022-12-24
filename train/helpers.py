import os


def get_all_local_labels():
    label_names = []
    for file_name in os.listdir("raw_data"):
        if file_name[0] == ".": continue

        label_name = file_name.split(".")[0]
        label_names.append(label_name)

    return label_names
