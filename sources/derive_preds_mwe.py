import copy
from time import time

import pandas as pd
import string

import time


def derive_preds_mwe(stimuli, external_data):
    """Derive predictors for multi-word expressions, from either behavioural norms/lexical resources, 
    or context-independent models.

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
    predictor_table : DataFrame, shape (n_stimuli, 9 * n_cols_for_predictors)
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
                
        
        # check whether the first target is covered by the external dataset                
        try:
                        
            curr_first_target_pos = word_list.index(stimuli['token'][i].lower().split(' ')[0])

        except:

            curr_first_target_pos = -1          
            
            
        # check whether the second target is covered by the external dataset                
        try:
                        
            curr_second_target_pos = word_list.index(stimuli['token'][i].lower().split(' ')[1])

        except:

            curr_second_target_pos = -1          
        
        
        # derive target predictors as follows: if the first or second targets are covered by the external dataset, 
        # then use the corresponding features or vector dimensions, otherwise use the average values of the features 
        # or vector dimensions, computed over all the words in the external dataset       
        if curr_first_target_pos != -1:

            curr_first_target_table = external_data.iloc[[curr_first_target_pos]].drop(['Word'], axis=1).reset_index(drop=True)

        else:

            curr_first_target_table = copy.deepcopy(pred_mean)
            
            
        if curr_second_target_pos != -1:

            curr_second_target_table = external_data.iloc[[curr_second_target_pos]].drop(['Word'], axis=1).reset_index(drop=True)

        else:

            curr_second_target_table = copy.deepcopy(pred_mean)
            
            
        curr_context_table = pd.DataFrame()
        curr_abs_diff_table_first = pd.DataFrame()
        curr_abs_diff_table_second = pd.DataFrame()
        curr_abs_diff_table_targets = pd.DataFrame()
        curr_prod_table_first = pd.DataFrame()
        curr_prod_table_second = pd.DataFrame()
        curr_prod_table_targets = pd.DataFrame()
                 
                 
        try:
        
            pos_in_sent_first = curr_context_words.index(stimuli['token'][i].lower().split(' ')[0])
            
        except:
        
            pos_in_sent_first = -1   
        
        
        try:
        
            pos_in_sent_second = curr_context_words.index(stimuli['token'][i].lower().split(' ')[1])
            
        except:
        
            pos_in_sent_second = -1
        
        
        context_pos = []
        n_missing_words = 0
        
        # iterate through the context words and collect the features or vector dimensions corresponding to each word                
        for j in range(len(curr_context_words)):

            if j != pos_in_sent_first and j != pos_in_sent_second:

                try:

                    curr_context_pos = word_list.index(curr_context_words[j])                    
                    context_pos.append(curr_context_pos)

                except:

                    n_missing_words += 1                 
        
        if len(context_pos) > 0:

            curr_context_table = (external_data.iloc[context_pos,1:].sum() + n_missing_words * pred_mean) / (len(context_pos) + n_missing_words)
            
        else:
        
            curr_context_table = copy.deepcopy(pred_mean)

        # derive absolute difference predictors for the first target as follows: use the absolute difference between the first target and context predictors
        curr_abs_diff_table_first = curr_first_target_table.subtract(curr_context_table).abs()
        
        # derive absolute difference predictors for the second target as follows: use the absolute difference between the second target and context predictors
        curr_abs_diff_table_second = curr_second_target_table.subtract(curr_context_table).abs()
        
        # derive absolute difference predictors for the two targets: use the absolute difference between the first target and second target predictors
        curr_abs_diff_table_targets = curr_first_target_table.subtract(curr_second_target_table).abs()

        # derive product predictors for the first target as follows: use the product between the first target and context predictors
        curr_prod_table_first = curr_first_target_table.multiply(curr_context_table)

        # derive product predictors for the second target as follows: use the product between the second target and context predictors
        curr_prod_table_second = curr_second_target_table.multiply(curr_context_table)
        
        # derive product predictors for the two targets as follows: use the product between the first target and second target predictors
        curr_prod_table_targets = curr_first_target_table.multiply(curr_second_target_table)
        

        # assign proper names to the target, context, absolute difference, and product predictors               
        curr_first_target_table.columns = ['First_Target_' + col for col in curr_first_target_table.columns]
        curr_second_target_table.columns = ['Second_Target_' + col for col in curr_second_target_table.columns]
        curr_context_table.columns = ['Context_' + col for col in curr_context_table.columns]
        
        column_names_interact = external_data.columns[1:]

        curr_abs_diff_table_first.columns = ['Abs_Diff_First_' + column_names_interact[j] for j in range(len(column_names_interact))]
        curr_abs_diff_table_second.columns = ['Abs_Diff_Second_' + column_names_interact[j] for j in range(len(column_names_interact))]
        curr_abs_diff_table_targets.columns = ['Abs_Diff_Targets_' + column_names_interact[j] for j in range(len(column_names_interact))]
        
        curr_prod_table_first.columns = ['Prod_First_' + column_names_interact[j] for j in range(len(column_names_interact))]
        curr_prod_table_second.columns = ['Prod_Second_' + column_names_interact[j] for j in range(len(column_names_interact))]
        curr_prod_table_targets.columns = ['Prod_Targets_' + column_names_interact[j] for j in range(len(column_names_interact))]        


        # combine all predictors
        predictor_table = pd.concat([predictor_table, pd.concat([curr_first_target_table, curr_second_target_table, curr_context_table, 
        curr_abs_diff_table_first, curr_abs_diff_table_second, curr_abs_diff_table_targets, 
        curr_prod_table_first, curr_prod_table_second, curr_prod_table_targets], axis=1)], axis=0)   
    
    return predictor_table


