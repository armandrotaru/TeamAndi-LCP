import copy
from time import time

import pandas as pd
import string

import time


def derive_preds_sw(stimuli, external_data):
    """Derive predictors from single words, from either behavioural norms/lexical resources, or context-independent models.

    Parameters
    ----------
    stimuli : DataFrame, shape (n_stimuli, n_cols_stimuli)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_cols_stimuli is the number of columns. The dataset must include at least the columns
        'token' and 'sentence'.

    external_data : DataFrame, shape (n_words, n_cols_for_predictors + 1)
        Behavioural norm or context-independent model. The data has shape (n_words, n_cols), where n_words is
        the number of words, and n_cols_for_predictors is the number of features or vector dimensions. The
        first column ('Word') contains the words, while the other columns contain the features or vector
        dimensions.
        
    Returns
    -------
    predictor_table : DataFrame, shape (n_stimuli, 4 * n_cols_for_predictors)
        Predictors derived from the behavioural norm or context-independent model, where n_stimuli is the
        number of stimuli, and n_cols_for_predictors is the number of features or vector dimensions.
    """       

    pred_mean = pd.DataFrame([list(external_data.iloc[:,1:].mean())], columns=external_data.columns[1:])    
    
    predictor_table = pd.DataFrame()

    word_list = list(external_data['Word'])

    # iterate through the stimuli
    for i in range(stimuli.shape[0]):
            
        cleaned_sentence = ''.join([ch for ch in stimuli['sentence'][i].lower().strip() if ch not in string.punctuation])
        curr_context_words = cleaned_sentence.split(' ')
        
        # check whether the target is covered by the external dataset                
        try:
                        
            curr_target_pos = word_list.index(stimuli['token'][i].lower())

        except:

            curr_target_pos = -1          
        
        
        # derive target predictors as follows: if the target is covered by the external dataset, then use the
        # corresponding features or vector dimensions, otherwise use the average values of the features or
        # vector dimensions, computed over all the words in the external dataset       
        if curr_target_pos != -1:

            curr_target_table = external_data.iloc[[curr_target_pos]].drop(['Word'], axis=1).reset_index(drop=True)

        else:

            curr_target_table = copy.deepcopy(pred_mean)
            
        curr_context_table = pd.DataFrame()
        curr_abs_diff_table = pd.DataFrame()
        curr_prod_table = pd.DataFrame()
        
        try:
        
            pos_in_sent = curr_context_words.index(stimuli['token'][i].lower())
            
        except:
        
            pos_in_sent = -1   
        
        
        context_pos = []
        n_missing_words = 0
        
        # iterate through the context words and collect the features or vector dimensions corresponding to each word                
        for j in range(len(curr_context_words)):

            if j != pos_in_sent:

                try:

                    curr_context_pos = word_list.index(curr_context_words[j])                    
                    context_pos.append(curr_context_pos)

                except:

                    n_missing_words += 1                 
        
        if len(context_pos) > 0:

            curr_context_table = (external_data.iloc[context_pos,1:].sum() + n_missing_words * pred_mean) / (len(context_pos) + n_missing_words)
            
        else:
        
            curr_context_table = copy.deepcopy(pred_mean)

        # derive absolute difference predictors as follows: use the absolute difference between the target and context predictors
        abs_diff_values_table = curr_target_table.subtract(curr_context_table).abs()

        # derive product predictors as follows: use the product between the target and context predictors
        prod_values_table = curr_target_table.multiply(curr_context_table)

        # assign proper names to the target, context, absolute difference, and product predictors
        column_names_interact = external_data.columns[1:]

        for j in range(len(column_names_interact)):

            curr_abs_diff_table['Abs_Diff_' + column_names_interact[j]] = abs_diff_values_table.iloc[:,j]
            curr_prod_table['Prod_' + column_names_interact[j]] = prod_values_table.iloc[:,j]

        curr_target_table.columns = ['Target_' + col for col in curr_target_table.columns]
        curr_context_table.columns = ['Context_' + col for col in curr_context_table.columns]

        # combine all predictors
        predictor_table = pd.concat([predictor_table, pd.concat([curr_target_table, curr_context_table, curr_abs_diff_table, curr_prod_table], axis=1)], axis=0)  
        
    return predictor_table


