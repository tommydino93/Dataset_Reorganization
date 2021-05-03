import os
import re
import random
import math
from shutil import copyfile
import nibabel as nib
import numpy as np

"""
Created on May 3, 2021

@author: Tommaso Di Noto

This script converts a BIDS dataset (https://bids.neuroimaging.io/) to the dataset format required to run nnUNet (https://github.com/MIC-DKFZ/nnUNet)
"""


def round_half_up(n, decimals=0):
    """This function rounds to the nearest integer number (e.g 2.4 becomes 2.0 and 2.6 becomes 3);
     in case of tie, it rounds up (e.g. 1.5 becomes 2.0 and not 1.0)
    Args:
        n (float): number to round
        decimals (int): number of decimal figures that we want to keep; defaults to zero
    """
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


def split_train_test(bids_path, regexp_sub, ext_gz, percentage=0.8):
    """This function splits the tof-mra volumes into two sub-lists: one for training subjects and one for test subjects
    Args:
        bids_path (str): path to BIDS dataset
        regexp_sub (_sre.SRE_Pattern): template match
        ext_gz (str): specific file extension to retain
        percentage (float): percentage of subjects to keep for training
    Returns:
        train_subs (list): it contains the training subjects
        test_subs (list): it contains the test subjects
    """
    all_tof_volumes = []
    for subdir, dirs, files in os.walk(bids_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            # only retain paths of TOF-MRA volumes
            if regexp_sub.search(file) and ext == ext_gz and "angio" in file and "derivatives" not in subdir:
                all_tof_volumes.append(file)

    nb_train_subs = int(round_half_up(percentage*len(all_tof_volumes)))  # count number of samples that correspond to the specified percentage
    train_subs = random.sample(all_tof_volumes, nb_train_subs)  # sample training cases
    test_subs = [item for item in all_tof_volumes if item not in train_subs]  # extract remaining test cases

    return train_subs, test_subs


def load_nifti(input_path):
    """This function loads a nii.gz from an input path as nib object, converts it to numpy and returns both, together with the affine matrix
    Args:
        input_path (str): path to nifti input file, usually saved as nii.gz
    Returns:
        nii_obj (Nifti1Image): input nibabel image
        nii_volume_ (np.array): input image as numpy array
        aff_matrix (np.array): affine matrix associated with input volume
    """
    nii_obj = nib.load(input_path)  # load nii file as nibabel object
    nii_volume_ = np.asanyarray(nii_obj.dataobj)  # convert from nibabel object to numpy array
    if len(nii_volume_.shape) > 3:  # if input volume has more than 3 dimensions
        nii_volume_ = np.squeeze(nii_volume_, axis=3)  # only keep first 3 dims; e.g. drop fourth dim (time dimension) which is useless in our case
    aff_matrix = nii_obj.affine  # extract and save affine matrix
    return nii_obj, nii_volume_, aff_matrix


def merge_aneurysms_into_unique_volume(label_folder, ext_gz, tof_mra_path):
    """This function extracts the label volume (if present) and merges multiple aneurysm labels into one unique label map
    Args:
        label_folder (str): path to folder where the aneurysm labels are stored
        ext_gz (str): file extension to match
        tof_mra_path (str): path to tof-mra volume
    Returns:
        mask_obj (nib.Nifti1Image): nibabel object containing the label volume
    """
    lesions = []
    for files in os.listdir(label_folder):
        ext = os.path.splitext(files)[-1].lower()  # get the file extension
        if "Lesion" in files and ext == ext_gz:
            lesions.append(files)

    assert len(lesions) <= 4, "In the Lausanne dataset, a patient can have maximum 4 aneurysms"
    _, tof_mra_volume, tof_mra_aff = load_nifti(tof_mra_path)
    
    # if list is not empty (i.e. subject has one or more aneurysm(s))
    if lesions:
        mask = np.zeros(tof_mra_volume.shape, dtype=int)  # initialize empty np.array with same shape as bet-tof volume
        for aneur_path in lesions:  # loop over aneurysm(s) found for this patient
            _, aneurysm_volume, _ = load_nifti(os.path.join(label_folder, aneur_path))
            if not np.array_equal(aneurysm_volume, aneurysm_volume.astype(bool)) or np.sum(aneurysm_volume) == 0:
                raise ValueError("Mask must be binary and non-empty")

            assert aneurysm_volume.shape == mask.shape, "The two volumes must have same shape"
            non_zero_coords = np.nonzero(aneurysm_volume)  # type: tuple # find coords of non-zero voxels
            mask[non_zero_coords] = 1  # assign 1 to non-zero coordinates
            if len(lesions) == 1:  # if there is only one aneurysm
                assert np.count_nonzero(mask) == np.count_nonzero(aneurysm_volume)

        # check that final mask is binary and non-empty
        if not np.array_equal(mask, mask.astype(bool)) or np.sum(mask) == 0:
            raise ValueError("Mask must be binary and non-empty")
        mask_obj = nib.Nifti1Image(mask, affine=tof_mra_aff)  # convert to nibabel object

    else:
        mask = np.zeros(tof_mra_volume.shape, dtype=int)  # type: np.ndarray # initialize empty np.array with same shape as bet-tof volume
        mask_obj = nib.Nifti1Image(mask, affine=tof_mra_aff)

    return mask_obj


def bids2nnunet(bids_path, nnunet_path):
    training_images_path = os.path.join(nnunet_path, "imagesTr")
    if not os.path.exists(training_images_path):
        os.makedirs(training_images_path)

    training_labels_path = os.path.join(nnunet_path, "labelsTr")
    if not os.path.exists(training_labels_path):
        os.makedirs(training_labels_path)

    test_images_path = os.path.join(nnunet_path, "imagesTs")
    if not os.path.exists(test_images_path):
        os.makedirs(test_images_path)
    
    regexp_sub = re.compile(r'sub')  # create a substring template to match
    ext_gz = '.gz'  # type: str # set zipped files extension
    cnt = 0  # type: int # dummy counter

    train_subs, test_subs = split_train_test(bids_path, regexp_sub, ext_gz, percentage=0.8)

    for subdir, dirs, files in os.walk(bids_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            # only retain paths of TOF-MRA volumes
            if regexp_sub.search(file) and ext == ext_gz and "angio" in file and "derivatives" not in subdir:
                cnt += 1
                sub = re.findall(r"sub-\d+", subdir)[0]
                ses = re.findall(r"ses-\d+", subdir)[0]
                print("{}) Copying files for {}_{}".format(cnt, sub, ses))

                # if we are dealing with a training subject, we save both the tof-mra volume and the labels
                if file in train_subs:
                    # we only retain the TOF-MRA modality which is assigned the identifier 0000
                    tof_mra_path = os.path.join(subdir, file)
                    volume_filename = "angio_{}_0000.nii.gz".format(str(cnt).zfill(3))
                    if not os.path.exists(os.path.join(training_images_path, volume_filename)):
                        copyfile(tof_mra_path, os.path.join(training_images_path, volume_filename))

                    # save label
                    label_folder = os.path.join(bids_path, "derivatives/manual_masks", sub, ses, "anat")
                    label_volume_obj = merge_aneurysms_into_unique_volume(label_folder, ext_gz, tof_mra_path)
                    label_filename = "angio_{}.nii.gz".format(str(cnt).zfill(3))
                    if not os.path.exists(os.path.join(training_labels_path, label_filename)):
                        nib.save(label_volume_obj, os.path.join(training_labels_path, label_filename))

                # if instead we are dealing with a test subject, we only save the tof-mra volume
                elif file in test_subs:
                    tof_mra_path = os.path.join(subdir, file)
                    volume_filename = "angio_{}_0000.nii.gz".format(str(cnt).zfill(3))
                    if not os.path.exists(os.path.join(test_images_path, volume_filename)):
                        copyfile(tof_mra_path, os.path.join(test_images_path, volume_filename))
                else:
                    raise ValueError("Subject should either be in train list or test list")


def main():
    bids_ds_path = "/media/newuser/7EF9B56B27259371/Aneurysm_Data_Set/29-BIDS_Aneurysm_dataset_Apr_06_2021/"  # path to BIDS dataset folder
    nnunet_ds_path = "/media/newuser/7EF9B56B27259371/Aneurysm_Data_Set/32-nnUNet_aneurysms/nnUNet_raw_data/Task101_AneurysmDetectionMRA/"  # path to nnUNet dataset folder
    bids2nnunet(bids_ds_path, nnunet_ds_path)


if __name__ == '__main__':
    main()
