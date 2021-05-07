import os
from typing import Tuple
import numpy as np
import json
import re

"""
Created on May 3, 2021

@author: Tommaso Di Noto

This script generates the file "dataset.json" required to run nnUNet (https://github.com/MIC-DKFZ/nnUNet)
"""


def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def get_identifiers_modified(folder):
    subs = []
    for files in os.listdir(folder):
        sub_id = re.findall(r"angio_\d+", files)[0]
        subs.append(files)

    subs = [item[:-7] for item in subs]
    subs_np = np.asarray(subs)
    return subs_np


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_modified(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {'name': dataset_name,
                 'description': dataset_description,
                 'tensorImageSize': "4D",
                 'reference': dataset_reference,
                 'licence': license,
                 'release': dataset_release,
                 'modality': {str(i): modalities[i] for i in range(len(modalities))},
                 'labels': {str(i): labels[i] for i in labels.keys()},
                 'numTraining': len(train_identifiers),
                 'numTest': len(test_identifiers),
                 'training': [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_identifiers],
                 'test': ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]}

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))


def main():
    generate_dataset_json("/media/newuser/7EF9B56B27259371/Aneurysm_Data_Set/32-nnUNet_aneurysms/nnUNet_raw_data/Task101_AneurysmDetectionMRA/dataset.json",
                          "/media/newuser/7EF9B56B27259371/Aneurysm_Data_Set/32-nnUNet_aneurysms/nnUNet_raw_data/Task101_AneurysmDetectionMRA/labelsTr",
                          "/media/newuser/7EF9B56B27259371/Aneurysm_Data_Set/32-nnUNet_aneurysms/nnUNet_raw_data/Task101_AneurysmDetectionMRA/imagesTs",
                          ("TOF-MRA",),
                          {0: 'background', 1: 'aneurysm'},
                          "Lausanne_TOF-MRA_Aneurysms",
                          license="CC BY-SA 4.0",
                          dataset_description="Dataset of TOF-MRA volumes for the detection and segmentation of unruptured intracranial aneurysms.")


if __name__ == '__main__':
    main()
