import os
import pandas as pd
import numpy as np

def prepare_fd(dataset_dir):
    """
    A utility function to grab motion correction data for all subjects,
    compute framewise displacement, load data into pandas datadrames,
    and return the dataframes and other lists to the plotting scripts.
    
    Parameters
    ----------
    dataset_dir: the absolute path of the studyforrest subdataset where
                 the 3T audiovisual movie data are located

    Returns
    -------
    participants: list os subject identifiers
    column_names: list of column_names for subs and runs
    df_subs: a pandas dataframe with a column per subject, where all FD
             values over all runs constitute the rows
    df_subsruns: a pandas dataframe with a column per subject+run, where
                 the FD values for the specific subject and run
                 constitute the rows
    """
    
    participants = [f"sub-{i:02d}" for i in range(1,21)]
    runs = [f"run-{i}" for i in range(1,9)]
    column_names = []
    df_subs = pd.DataFrame(columns=participants)

    for sub in participants:
        for run in runs:
            marker = sub + '_' + run
            moco_fn = os.path.join(dataset_dir, sub, 'in_bold3Tp2', sub + '_task-avmovie_' + run + '_bold_mcparams.txt')
            if os.path.exists(moco_fn):
                column_names.append(marker)
    
    df_subsruns = pd.DataFrame(columns=column_names)

    for sub in participants:
        fd_all = []
        for run in runs:
            marker = sub + '_' + run
            # print(marker)
            moco_fn = os.path.join(dataset_dir, sub, 'in_bold3Tp2', sub + '_task-avmovie_' + run + '_bold_mcparams.txt')
            try:
                moco_params = pd.read_csv(moco_fn, delim_whitespace=True).to_numpy()
            except FileNotFoundError:
                # doesn't exist. do nothing.
                txt = 'File does not exist: ' + sub + '_ses-forrestgump_acq-dico_' + run + '_bold_moco.txt; skipping for now...'
                # print(txt)
            else:
                translations_mm = moco_params[:, 3:6] # last 3 columns
                rotations_mm = 50*moco_params[:, 0:3] # already in radians, first 3 columns
                moco_params_mm = np.concatenate((translations_mm, rotations_mm), axis=1)
                # Calculate FD
                fd = np.sum(np.abs(np.diff(moco_params_mm, axis=0)), axis=1)
                fd = np.insert(fd, 0, 0)
                fd_list = fd.tolist()
                fd_all = fd_all + fd_list
                # Add to dataframes
                df_subsruns.loc[:,marker] = pd.Series(fd_list)
        
        df_subs.loc[:,sub] = pd.Series(fd_all)

    return participants, column_names, df_subs, df_subsruns