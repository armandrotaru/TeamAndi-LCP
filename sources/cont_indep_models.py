import numpy as np
import pandas as pd
from time import time

from sources.derive_preds_sw import derive_preds_sw
from sources.derive_preds_mwe import derive_preds_mwe



def load_models(cont_indep_model_names, cont_indep_model_filenames, verbose):
    """"Load and process the context-independent models.

    The context-independent models (i.e., embeddings) are read from file, which is assumed to have no header.
    The first column in each file must contains the words, while the other columns must contain the vector dimensions.

    Parameters
    ----------
    cont_indep_model_names : str array, shape (n_cont_indep_models)
        Names of the context-independent models, where n_cont_indep_models is the number of models.

    cont_indep_model_filenames : str array, shape (n_cont_indep_models)
        Names of the files storing the context-independent models (i.e., embeddings), where
        n_cont_indep_models is the number of models.

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    cont_indep_models : DataFrame array, shape (n_cont_indep_models)
        Context-independent models, where n_cont_indep_models is the number of context-independent models.
        Each model has shape (n_words, n_dims+1), where n_words is the number of words, and n_dims is the
        number of vector dimensions, both of which are specific to each model. For each model, the first
        column ('Word') contains the words, while the other columns contain the vector dimensions.
    """

    cont_indep_models = []

    # iterate through all the models
    for curr_cont_indep_model_filename, curr_cont_indep_model_name in zip(cont_indep_model_filenames, cont_indep_model_names):

        if verbose:

            start_time = time()

        # load each model from file
        curr_cont_indep_model = pd.read_csv(curr_cont_indep_model_filename, header=None)

        # convert numeric values from 64 to 32 bit, in order to save memory and computation time
        curr_cont_indep_model.iloc[:,1:] = curr_cont_indep_model.iloc[:,1:].astype(np.float32)

        # generate column names for the current model
        curr_cont_indep_model_col_names = ['Word']

        for i in range(len(curr_cont_indep_model.columns)-1):

            curr_cont_indep_model_col_names.append(curr_cont_indep_model_name + '_Dim_' + str(i+1))

        curr_cont_indep_model.columns = curr_cont_indep_model_col_names

        # save each loaded model
        cont_indep_models.append(curr_cont_indep_model)

        if verbose:

            finish_time = time()

            run_duration = int(finish_time - start_time)

            # notify the user of the successful completion of the task, together with its duration
            print('({}s) Loaded {} model'.format(run_duration, curr_cont_indep_model_name))

    return cont_indep_models



def generate_preds(stimuli, cont_indep_models, cont_indep_model_names, pred_names, use_single_words, verbose):
    """Generate predictors from the context-independent models.

    The predictors are derived from the previously loaded models.

    Parameters
    ----------
    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns 'token' and 'sentence'.

    cont_indep_models : DataFrame array, shape (n_cont_indep_models)
        Context-independent models, where n_cont_indep_models is the number of context-independent models.
        Each model has shape (n_words, n_dims+1), where n_words is the number of words, and n_dims is the
        number of vector dimensions, both of which are specific to each model. For each model, the first
        column ('Word') contains the words, while the other columns contain the vector dimensions.

    cont_indep_model_names : str array, shape (n_cont_indep_models)
        Names of the individual context-independent models, where n_cont_indep_models is the number of
        individual models.

    pred_names : str array, shape (n_norms_and_models_sel)
        Names of the behavioural norms and distributional models selected by the user, where 
        n_norms_and_models_sel is the number of selected norms and models.

    use_single_words : bool
        Whether to generate predictors for single words, or multi-word expressions.

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    preds_cont_indep_models : DataFrame array, shape (n_cont_indep_models_sel)
        Predictors derived from the context-independent models selected by the user, where
        n_cont_indep_models_sel is the number of such models. Each set of predictors is of shape
        (n_stimuli, n_preds), where n_stimuli is the number of words, and n_preds is the number of predictors.
    """
    
    preds_cont_indep_models = []

    if len(cont_indep_models) > 0:

        # iterate through all the selected models
        for curr_pred_name in pred_names:

            if curr_pred_name in cont_indep_model_names:
                
                if verbose:

                    start_time = time()
                
                curr_cont_indep_model = cont_indep_models[cont_indep_model_names.index(curr_pred_name)]

                # derive predictors, using the current model
                if use_single_words:
                
                    curr_preds = derive_preds_sw(stimuli['X'], curr_cont_indep_model)
                    
                else:
                
                    curr_preds = derive_preds_mwe(stimuli['X'], curr_cont_indep_model)

                preds_cont_indep_models.append(curr_preds)
                
                if verbose:
        
                    finish_time = time()

                    run_duration = int(finish_time - start_time)

                    # notify the user of the successful completion of the task, together with its duration
                    print('({}s) Generated predictors for {}'.format(run_duration, curr_pred_name))
                        
    return preds_cont_indep_models
