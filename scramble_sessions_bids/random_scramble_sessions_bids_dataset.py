#!/usr/bin/env python
"""
This script randomly scrambles the sessions in a BIDS dataset (https://bids.neuroimaging.io/) by a range of +/- [0, days_to_shift]
"""

import os
import random
import datetime
import pandas as pd
import json


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "Tommaso.Di-Noto@chuv.ch"
__status__ = "Prototype"


def keep_only_digits(input_string: str) -> str:
    """This function takes as input a string and returns the same string but only containing digits
    Args:
        input_string (str): the input string from which we want to remove non-digit characters
    Returns:
        output_string (str): the output string that only contains digit characters
    """
    numeric_filter = filter(str.isdigit, input_string)
    output_string = "".join(numeric_filter)

    return output_string


def fifty_fifty() -> bool:
    """This function returns True 50% of the times."""
    if random.random() < .5:
        return True
    return False


def dir_contains_dirs(input_path: str) -> bool:
    """This function checks whether the input_path contains any directory.
    Args:
        input_path (str): path where we search for any directory
    Returns:
        contains_any_dir (bool): True if there is at least one directory inside the input_path; False otherwise
    """
    contains_any_dir = False  # initialize as False
    for file in os.listdir(input_path):
        if os.path.isdir(os.path.join(input_path, file)):
            contains_any_dir = True

    return contains_any_dir


def rename_file(df_mapping: pd.DataFrame, input_dir: str, sub: str, ses: str, file: str) -> (str, str):
    """This function renames a the input file using the session mapping.
    Args:
        df_mapping (pd.DataFrame): mapping that links old sessions with scrambled sessions
        input_dir (str): path where the file to rename is located
        sub (str): subject ID
        ses (str): old session (the one that was scrambled)
        file (str): old file name that will be renamed
    Returns:
        scrambled_ses (str): scrambled sessions for this sub-ses combination
        new_filename (str): name of modified file
    """
    row_of_interest = df_mapping[(df_mapping["sub"] == sub) & (df_mapping["orig_ses"] == ses)]
    assert row_of_interest.shape[0] == 1, "More (or less) than one row matched; there should be only one sub-ses match."
    scrambled_ses = row_of_interest["scrambled_ses"].values[0]  # extract scrambled ses
    new_filename = file.replace(ses, scrambled_ses)
    # rename file
    os.rename(os.path.join(input_dir, file), os.path.join(input_dir, new_filename))
    # print("renamed", file, "into", new_filename)
    return scrambled_ses, new_filename


def scramble_sessions_in_files(input_dir: str, df_mapping: pd.DataFrame, sub: str, ses: str) -> str:
    """This function loops over all the files in the input_dir anc changes the old session with the scrambled one.
    Args:
        input_dir (str): path to folder where the files are located
        df_mapping (pd.DataFrame): mapping that links old sessions with scrambled sessions
        sub (str): subject ID
        ses (str): old session (the one that was scrambled)
    Returns:
        scrambled_ses (str): scrambled session. It's initialized to None, but it should be modified
    """
    # define extensions that we expect
    ext_gz = ".gz"  # type: str
    ext_json = ".json"  # type: str
    ext_csv = ".csv"  # type: str
    ext_mat = ".mat"  # type: str

    scrambled_ses = None  # initialize to None; it should be modified inside the for loop

    for file in os.listdir(input_dir):
        # extract file extension
        ext = os.path.splitext(file)[-1].lower()  # type: str

        # if we are dealing with a .gz file
        if ext == ext_gz:
            scrambled_ses, _ = rename_file(df_mapping, input_dir, sub, ses, file)

        # if we are dealing with a .json file
        elif ext == ext_json:
            scrambled_ses, new_filename = rename_file(df_mapping, input_dir, sub, ses, file)

            # Modify also the content of the JSON file
            f = open(os.path.join(input_dir, new_filename), 'r')
            json_dict = json.load(f)  # load JSON object as a dictionary
            for key, value in json_dict.items():  # loop over dictionary
                if isinstance(key, str):
                    if ses in key:
                        raise ValueError("There should not be the session number in the keys of the json dict. Check {}_{}".format(sub, ses))
                if isinstance(value, str):
                    if ses in value:  # if the dict value contains the old session
                        new_value_name = value.replace(ses, scrambled_ses)
                        json_dict[key] = new_value_name
                        # save modified json file overwriting existing one
                        with open(os.path.join(input_dir, new_filename), "w") as outfile:
                            json.dump(json_dict, outfile)
                        # print("changed in json file: {} for {}".format(value, new_value_name))

        # if we are dealing with a .csv file
        elif ext == ext_csv:
            scrambled_ses, _ = rename_file(df_mapping, input_dir, sub, ses, file)

        # if we are dealing with a .mat file
        elif ext == ext_mat:
            scrambled_ses, _ = rename_file(df_mapping, input_dir, sub, ses, file)
        
        else:
            raise ValueError("Extension {} is not allowed for file {}; only .gz, .json, .csv and .mat are expected.".format(ext, file))

    return scrambled_ses


def scramble_sessions_across_subs(parent_dir: str, df_mapping: pd.DataFrame) -> None:
    """This function loops over all the subjects in parent_dir and recursively changes the old session with the scrambled session where appropriate.
    Args:
        parent_dir (str): directory in which subjects are located
        df_mapping (pd.DataFrame): mapping that links old sessions with scrambled sessions
    Returns:
        None
    """
    for sub in os.listdir(parent_dir):  # loop over files
        if os.path.isdir(os.path.join(parent_dir, sub)) and "sub-" in sub:  # filter only for subject folders
            print("--- Changing {}".format(sub))
            for ses in os.listdir(os.path.join(parent_dir, sub)):

                # if there is at least one directory inside the session folder (e.g. the anat/ folder)
                scrambled_ses = None  # initialize to None; it should be modified either in the "if" or in the "else"
                if dir_contains_dirs(os.path.join(parent_dir, sub, ses)):
                    for img_type in os.listdir(os.path.join(parent_dir, sub, ses)):  # img_type can be for instance anat/, func/, dwi/
                        scrambled_ses = scramble_sessions_in_files(os.path.join(parent_dir, sub, ses, img_type), df_mapping, sub, ses)

                # if instead there are no directories (i.e. there are only files)
                else:
                    scrambled_ses = scramble_sessions_in_files(os.path.join(parent_dir, sub, ses), df_mapping, sub, ses)

                # change name of session folder
                assert scrambled_ses, "scrambled_ses should not be None; check {}_{}".format(sub, ses)
                os.rename(os.path.join(parent_dir, sub, ses), os.path.join(parent_dir, sub, scrambled_ses))
                # print("Changed name of ses folder from {} to {}\n".format(ses, scrambled_ses))


def create_mapping_old_ses_2_scrambled_ses(bids_dir: str, days_to_shift: int) -> pd.DataFrame:
    """This function loops over all sub-ses combinations and for each of those it creates a corresponding scrambled session in the range +- [0, days_to_shift].
    When multiple sessions are present, the function applies the same shift to all sessions so that the order in time is preserved.
    Args:
        bids_dir (str): path to the BIDS dataset
        days_to_shift (int): number of days to shift
    Returns:
        df_mapping (pd.DataFrame): mapping that links old sessions with scrambled sessions
    """
    mapping_old_ses_new_ses = []  # type: list # used as mapping to keep track of the scrambles
    # create session mapping
    for sub in os.listdir(bids_dir):  # loop over files
        if os.path.isdir(os.path.join(bids_dir, sub)) and "sub-" in sub:  # filter only for subject folders

            # if subject has zero sessions, raise error
            if len(os.listdir(os.path.join(bids_dir, sub))) <= 0:
                raise ValueError("{} does not contain sessions; at least 1 session per subject is expected")

            # if subject has at least one session
            else:
                random_flag = fifty_fifty()  # randomly pick whether to add or subtract days; if flag is True we add, otherwise we subtract
                random_number_of_days = random.randrange(days_to_shift)  # type: int  # set random number of days to shift in the range [0, days_to_shift]

                # loop over sessions (there can be more than one)
                for ses in os.listdir(os.path.join(bids_dir, sub)):
                    ses_only_digits = keep_only_digits(ses)  # type: str # extract only digits
                    ses_datetime = datetime.datetime.strptime(ses_only_digits, '%Y%m%d')  # type: datetime.datetime # convert to datetime format

                    # if flag is True we add days, otherwise we subtract days
                    if random_flag:
                        scrambled_ses = ses_datetime + datetime.timedelta(days=random_number_of_days)  # type: datetime.datetime
                    else:
                        scrambled_ses = ses_datetime - datetime.timedelta(days=random_number_of_days)  # type: datetime.datetime

                    scrambled_ses_as_string = scrambled_ses.strftime('%Y%m%d')  # type: str # convert back to string
                    scrambled_ses_as_string = "ses-{}".format(scrambled_ses_as_string)

                    # group everything into one list
                    one_sub_ses = [sub, ses, scrambled_ses_as_string]
                    mapping_old_ses_new_ses.append(one_sub_ses)

    # convert from list of lists to dataframe
    df_mapping = pd.DataFrame(mapping_old_ses_new_ses, columns=['sub', 'orig_ses', 'scrambled_ses'])  # type: pd.DataFrame

    return df_mapping


def scramble_sessions_in_participants_tsv(bids_dir: str, df_mapping: pd.DataFrame) -> None:
    """This function changes the sessions (from original to scrambled) in the participant.tsv file of the BIDS dataset.
    Args:
        bids_dir (str): path to BIDS dataset
        df_mapping (pd.DataFrame): mapping that links old sessions with scrambled sessions
    Returns:
        None
    """
    assert os.path.exists(os.path.join(bids_dir, "participants.tsv")), "participants.tsv not found in main BIDS directory"
    df_participants = pd.read_csv(os.path.join(bids_dir, "participants.tsv"), sep='\t')

    for idx, row in df_participants.iterrows():  # loop over rows
        sub = row['participant_id']
        ses = row['exam_date']
        ses = "ses-{}".format(ses)
        # find corresponding row from mapping dataframe
        row_of_interest = df_mapping[(df_mapping["sub"] == sub) & (df_mapping["orig_ses"] == ses)]
        if row_of_interest.shape[0] == 1:
            # assert row_of_interest.shape[0] == 1, "More (or less) than one row matched; there should be only one sub-ses match."
            scrambled_ses = row_of_interest["scrambled_ses"].values[0]  # extract scrambled ses
            scrambled_ses_only_digits = int(keep_only_digits(scrambled_ses))
            # change session value in dataframe replacing old with scrambled
            df_participants.at[idx, 'exam_date'] = scrambled_ses_only_digits
        else:
            print("Warning: sub-ses match not found for {}_{}".format(sub, ses))

    # save modified dataframe to disk
    df_participants.to_csv(os.path.join(bids_dir, "participants.tsv"), sep='\t', index=False)


def scramble_sessions(bids_dir: str, out_dir: str, days_to_shift: int = 31) -> None:
    """This function loops over all directories and files of the input BIDS dataset (https://bids.neuroimaging.io/) and randomly scrambles
    the session dates in the range +- [0, days_to_shift]. If one subject has more than one session, the script ensures that even after the
    random scramble the order of the sequences in time does not change (e.g. if a subject has 2 sessions, one in March and one in April, even
    after the scramble the March session will be the first one).
    Args:
        bids_dir (str): path to BIDS dataset
        out_dir (str): output path where we save the session mapping (old session -> scrambled session)
        days_to_shift (int): we shift the dates; defaults to 31 days
    Returns:
        None
    """
    # set a random seed for reproducibility; comment this line to obtain a different random scramble at every run
    random.seed(123)

    # --------- 1) create a Dataframe that maps original sessions to scrambled sessions ---------
    print("\n------------------------------ Creating session mapping...")
    df_mapping = create_mapping_old_ses_2_scrambled_ses(bids_dir, days_to_shift)
    # save mapping to disk
    date = datetime.datetime.today().strftime('%b_%d_%Y')  # type: str # save today's date
    df_mapping.to_csv(os.path.join(out_dir, "mapping_old_ses2scrambled_ses_{}.csv".format(date)))
    print("\nMapping saved in {}".format(out_dir))

    # --------- 2) Scramble sessions in original (i.e. non-derivatives) data directory ---------
    print("\n------------------------------ Converting sessions for original data...")
    scramble_sessions_across_subs(bids_dir, df_mapping)

    # --------- 3) Scramble sessions in derivatives folder(s) ---------
    print("\n------------------------------ Converting sessions for derivatives data...")
    for derivatives_dirname in os.listdir(os.path.join(bids_dir, "derivatives")):  # we can have multiple derivatives folders; loop over all of them
        # if we are dealing with a proper BIDS dataset folder (i.e. if it contains the dataset_description file)
        if "dataset_description.json" in os.listdir(os.path.join(bids_dir, "derivatives", derivatives_dirname)):
            print("\n-------------- {}".format(derivatives_dirname))
            scramble_sessions_across_subs(os.path.join(bids_dir, "derivatives", derivatives_dirname), df_mapping)
        # if there is no dataset_description file but the folder is not empty, it means there are further subdirectories
        elif "dataset_description.json" not in os.listdir(os.path.join(bids_dir, "derivatives", derivatives_dirname)) and os.listdir(os.path.join(bids_dir, "derivatives", derivatives_dirname)):
            print("\n-------------- {}".format(derivatives_dirname))
            for derivatives_subdir in os.listdir(os.path.join(bids_dir, "derivatives", derivatives_dirname)):
                print("\n------- {}".format(derivatives_subdir))
                scramble_sessions_across_subs(os.path.join(bids_dir, "derivatives", derivatives_dirname, derivatives_subdir), df_mapping)
        # if there is no dataset_description and the derivatives folder is empty -> raise error
        else:
            raise IOError("Unexpected case; check folder {}".format(os.path.join(bids_dir, "derivatives", derivatives_dirname)))

    # change sessions in participants.tsv file
    scramble_sessions_in_participants_tsv(bids_dir, df_mapping)


def main():
    # define input args
    bids_dir = "/path/to/BIDS/dataset"  # path to BIDS dataset
    days_to_shift = 31  # type: int # we randomly shift the dates in the range of +- [0, days_to_shift]
    out_dir = "/path/to/output/dit"  # path where we save the session mapping (old session -> scrambled session)

    scramble_sessions(bids_dir, out_dir, days_to_shift)


if __name__ == '__main__':
    main()
