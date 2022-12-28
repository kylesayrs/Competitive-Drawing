import os


def get_all_local_labels():
    label_names = []
    for file_name in os.listdir("raw_data"):
        if file_name[0] == ".": continue

        label_name = file_name.split(".")[0]
        label_names.append(label_name)

    return label_names


class RandomResizePad(object):
    def __init__(self, size, scale):
        self.size = size
        self.scale = scale


    def __call__(self, sample):
        sample = F.to_pil_image(sample)
        crop_size = self.t.get_params(sample, self.scale, self.ratio)

        x_size = crop_size[2] - crop_size[0]
        y_size = crop_size[3] - crop_size[1]

        x_ratio = sample.size[0] / x_size
        y_ratio = sample.size[1] / y_size
        ratio = (x_ratio, y_ratio)

        output = F.crop(sample, *crop_size)
        output = F.resize(output, self.size)

        return ratio, output
