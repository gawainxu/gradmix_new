import numpy as np
import os
import scipy.io

import torchvision.transforms as transforms
from util import common_elements, TwoCropTransform
import torch.utils.data as data


class paderborn_machine(data.Dataset):
    # https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download
    # The dataset consists of 32 classes of health or damaged types (see self.class_names)
    # Each condition is measured under 4 conditions (see self.measure_conditions)

    #  Properly extracts the vibration signal from the MATLAB structure of Paderborn.
    #  The usual structure is: mat['FileName'][0,0]['Y'][0,0]['Data']
    #  current 1 data_array[0,0]['Y'][0,1]["Data"]
    #  current 2 data_array[0,0]['Y'][0,2]["Data"]
    #  vibration data_array[0,0]['Y'][0,6]["Data"]
    # TODO check the data length and frequency
    def __init__(self, root="../datasets/", train=True, transform=None, target_transform=None,
                 split_mode="none", win_len=0, step=100, classes=list(range(32)),
                 seperate_conditions=False, condition_name="", signal_name="current1"):
        self.class_names = ['K001', 'K002', 'K003', 'K004', 'K005', 'K006',
                            'KA01', 'KA03', 'KA04', 'KA05', 'KA06', 'KA07',
                            'KA08', 'KA09', 'KA15', 'KA16', 'KA22', 'KA30',
                            'KB23', 'KB24', 'KB27', 'KI01', 'KI03', 'KI04',
                            'KI05', 'KI07', 'KI08', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
        self.measure_conditions = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
        self.class_dict = {n: i for i, n in enumerate(self.class_names)}
        self.data = {n: {"N15_M07_F10": {"current1": [], "current2": [], "vibration": [], "label": i},
                         "N09_M07_F10": {"current1": [], "current2": [], "vibration": [], "label": i},
                         "N15_M01_F10": {"current1": [], "current2": [], "vibration": [], "label": i},
                         "N15_M07_F04": {"current1": [], "current2": [], "vibration": [], "label": i},}
                     for i, n in enumerate(self.class_names)}

        if train:
            self.root = os.path.join(root, "paderborn/train")
        else:
            self.root = os.path.join(root, "paderborn/test")
        self.transform = transform
        self.target_transform = target_transform
        self.win_len = win_len
        self.step = step
        self.classes = classes
        self.condition_name = condition_name
        self.signal_name = signal_name

        self.data_list = os.listdir(self.root)
        self.data_names = [s.split(".")[0] for s in self.data_list]

        # load and save the data in dict with the class names
        self.load_data()

        if split_mode == "window":
            self.split_data_window()

        self.output_seqs = []
        self.output_labels = []
        for class_name, class_index in self.class_dict.items():
            if class_index not in self.classes:
                continue
            class_data = self.data[class_name]
            class_label = self.data[class_name]
            if seperate_conditions:
                class_data = class_data[self.condition_name][self.signal_name]
                class_label = class_label[self.condition_name]["label"]
                self.output_seqs.extend(class_data)
                self.output_labels.extend([class_label] * len(class_data))
            else:
                for cn in self.measure_conditions:
                    condition_data = class_data[cn][self.signal_name]
                    condition_label = class_label[cn]["label"]
                    self.output_seqs.extend(condition_data)
                    self.output_labels.extend([condition_label] * len(condition_data))

    def load_data(self):

        for d in self.data_list:
            files_path = os.path.join(self.root, d)
            files_list = os.listdir(files_path)
            for fn in files_list:
                print(d, fn)
                if ".mat" not in fn:
                    continue
                file_path = os.path.join(files_path, fn)
                mat_data = scipy.io.loadmat(file_path, verify_compressed_data_integrity=False)
                key = fn.split(".")[0]
                data_array = mat_data[key]

                current1 = data_array[0,0]['Y'][0,1]["Data"]
                current2 = data_array[0,0]['Y'][0,2]["Data"]
                vibiration = data_array[0,0]['Y'][0,6]["Data"]

                for k in self.measure_conditions:
                    if k in key:
                        condition_key=k
                        break
                self.data[d][condition_key]["current1"].append(current1)
                self.data[d][condition_key]["current2"].append(current2)
                self.data[d][condition_key]["vibration"].append(vibiration)

    def split_data_window(self):

        for fn in self.class_names:
            for c in self.measure_conditions:
                currents1 = self.data[fn][c]["current1"]
                currents2 = self.data[fn][c]["current2"]
                vibration = self.data[fn][c]["vibration"]

                new_current1 = self.split_into_windows(currents1)
                new_current2 = self.split_into_windows(currents2)
                new_vibration = self.split_into_windows(vibration)

                self.data[fn][c]["current1"] = new_current1
                self.data[fn][c]["current2"] = new_current2
                self.data[fn][c]["vibration"] = new_vibration


    def split_into_windows(self, list_of_seqs):

        new_list_of_seqs = []
        for seq in list_of_seqs:
            seq = np.squeeze(seq)
            for i in range(0, len(seq), self.step):
                new_list_of_seqs.append(seq[i:i+self.win_len])

        return new_list_of_seqs


    def __getitem__(self, index):

        seq = self.output_seqs[index]
        label = self.output_labels[index]

        if self.transform is not None:
            seq = self.transform(seq)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return seq, label

    def __len__(self):

        return len(self.output_seqs)



if __name__ == "__main__":
    
    root_path = "../datasets/"
    train = True
    split_mode = "window"
    win_len = 4096
    step = 100
    classes = list(range(32))

    pd_ds = paderborn_machine(root=root_path, train=train, transform=None, target_transform=None,
                              split_mode=split_mode, win_len=win_len, step=step, classes=classes)
    print(len(pd_ds))


