import pandas as pd
from time import time

from sources.derive_preds_sw import derive_preds_sw
from sources.derive_preds_mwe import derive_preds_mwe



def load_norms(norm_names, norm_filenames, verbose):
    """Load and process the behavioural norms/lexical resources.

    The norms are read from file, which is assumed to have a header. The first column in each file ('Word')
    must contains the words, while the other columns must contain the features.

    Parameters
    ----------
    norm_names : str array, shape (n_behav_norms)
        Names of the behavioural norms, where n_behav_norms is the number of norms.

    norm_filenames : str array, shape (n_behav_norms)
        Names of the files storing the behavioural norms, where n_behav_norms is the number of norms.

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    behav_norms : DataFrame array, shape (n_behav_norms)
        Behavioural norms, where n_behav_norms is the number of behavioural norms. Each norm has shape
        (n_words, n_features + 1), where n_words is the number of words, and n_features is the number of
        features, both of which are specific to each norm. For each norm, the first column ('Word') contains
        the words, while the other columns contain the features.
    """

    behav_norms = []

    # iterate through all the norms
    for curr_norm_filename, curr_norm_name in zip(norm_filenames, norm_names):

        if verbose:

            start_time = time()

        # load each norm from file
        curr_norm = pd.read_csv(curr_norm_filename)
        

        # save each loaded norm
        behav_norms.append(curr_norm)

        if verbose:

            finish_time = time()

            run_duration = int(finish_time - start_time)

            # notify the user of the successful completion of the task, together with its duration
            print('({}s) Loaded {} norms'.format(run_duration, curr_norm_name))

    return behav_norms



def generate_preds(stimuli, behav_norms, norm_names, pred_names, use_single_words, verbose):
    """Generate predictors from the behavioural norms.

    The predictors are derived from the previously loaded norms.

    Parameters
    ----------
    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns 'token' and 'sentence'.

    behav_norms : DataFrame array, shape (n_behav_norms)
        Behavioural norms, where n_behav_norms is the number of behavioural norms. Each norm has shape
        (n_words, n_features + 1), where n_words is the number of words, and n_features is the number of
        features, both of which are specific to each norm. For each norm, the first column ('Word') contains
        the words, while the other columns contain the features.

    norm_names : str array, shape (n_behav_norms)
        Names of the behavioural norms, where n_behav_norms is the total number of norms.

    pred_names : str array, shape (n_norms_and_models_sel)
        Names of the behavioural norms and distributional models selected by the user, where 
        n_norms_and_models_sel is the number of selected norms and models.
        
    use_single_words : bool
        Whether to generate predictors for single words, or multi-word expressions.

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    preds_behav_norms : DataFrame array, shape (n_behav_norms_sel)
        Predictors derived from the behavioural norms selected by the user, where n_behav_norms_sel is the
        number of such norms. Each set of predictors is of shape (n_stimuli, n_preds), where n_stimuli is the
        number of stimuli, and n_preds is the number of predictors.
    """

    preds_behav_norms = []

    if len(behav_norms) > 0:

        # iterate through all the norms
        for curr_pred_name in pred_names:

            if curr_pred_name in norm_names:
                
                if verbose:

                    start_time = time()

                curr_norm = behav_norms[norm_names.index(curr_pred_name)]

                # derive predictors, using the current norm 
                if use_single_words:
                
                    curr_preds = derive_preds_sw(stimuli['X'], curr_norm)
                                    
                else:
                
                    curr_preds = derive_preds_mwe(stimuli['X'], curr_norm)
                
                preds_behav_norms.append(curr_preds)
                
                if verbose:
        
                    finish_time = time()

                    run_duration = int(finish_time - start_time)

                    # notify the user of the successful completion of the task, together with its duration
                    print('({}s) Generated predictors for {}'.format(run_duration, curr_pred_name))

    return preds_behav_norms


