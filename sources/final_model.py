import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

import sources.behav_norms as behav_norms
import sources.cont_indep_models as cont_indep_models
import sources.cont_dep_models as cont_dep_models
import sources.misc_utils as misc_utils



class LcpRegressor(RegressorMixin):
    """Regression model used by team Andi in the LCP challenge, part of SemEval 2021.

    Employs behavioural norms/lexical resources and distributional models, in order to predict subjective 
    complexity ratings for stimuli presented in a sentential context.

    Parameters
    ----------
    lambda_param : float, default=500
        Regularization strength used for the ridge regression.

    verbose : bool, default=True
        Whether to provide a step-by-step breakdown of the fitting, prediction, and scoring steps, together
        with the duration of each individual step.

    Attributes
    ----------
    behav_norms : DataFrame array, shape (n_behav_norms)
        Behavioural norms, where n_behav_norms is the number of behavioural norms. Each norm has shape
        (n_words, n_features + 1), where n_words is the number of words, and n_features is the number of
        features, both of which are specific to each norm. For each norm, the first column ('Word') contains
        the words, while the other columns contain the features.

    cont_indep_models : DataFrame array, shape (n_cont_indep_models)
        Context-independent models, where n_cont_indep_models is the number of context-independent models.
        Each model has shape (n_words, n_dims+1), where n_words is the number of words, and n_dims is the
        number of vector dimensions, both of which are specific to each model. For each model, the first
        column ('Word') contains the words, while the other columns contain the vector dimensions.

    cont_dep_models : DataFrame array, shape (n_cont_dep_models)
        Context-dependent models, where n_cont_dep_models is the number of context-dependent models. Each
        model has shape (n_words, n_dims+1), where n_words is the number of words, and n_dims is the number
        of vector dimensions, both of which are specific to each model. For each model, the first
        column ('Word') contains the words, while the other columns contain the vector dimensions.

    tokenizers : Tokenizer array, shape (n_cont_dep_models)
        Tokenizers corresponding to the context-dependent models, where n_cont_dep_models is the number of
        context-dependent models. The context-dependent models and the tokenizers are matched position-wise.

    pred_names : str array, shape (n_norms_and_models_sel)
        Names of the behavioural norms and distributional models selected by the user, where n_norms_and_models_sel 
        is the number of selected norms and models.

    preds_behav_norms : DataFrame array, shape (n_behav_norms_sel)
        Predictors derived from the behavioural norms selected by the user, where n_behav_norms_sel is the
        number of such norms. Each set of predictors is of shape (n_words, n_preds), where n_words is the
        number of words, and n_preds is the number of predictors.

    preds_cont_indep_models : DataFrame array, shape (n_cont_indep_models_sel)
        Predictors derived from the context-independent models selected by the user, where
        n_cont_indep_models_sel is the number of such models. Each set of predictors is of shape (n_words,
        n_preds), where n_words is the number of words, and n_preds is the number of predictors.

    preds_cont_dep_models : DataFrame array, shape (n_cont_dep_models_sel)
        Predictors derived from the context-dependent models selected by the user, where
        n_cont_dep_models_sel is the number of such models. Each set of predictors is of shape (n_words,
        n_preds), where n_words is the number of words, and n_preds is the number of predictors.

    use_single_words : bool
        Whether the targets consist of single words, or multi-word expressions.

    lambda_param : float, default=500
        Regularization strength used for the ridge regression.
        
    verbose : bool, default=True
        Whether to provide a step-by-step breakdown of the fitting, prediction, and scoring steps, together
        with the duration of each individual step.
    """



    def __init__(self, use_single_words=True, lambda_param=500, verbose=True):

        self.behav_norms = []
        self.cont_indep_models = []
        self.cont_dep_models = []

        self.tokenizers = []

        self.pred_names = []
        self.preds_behav_norms = []
        self.preds_cont_indep_models = []
        self.preds_cont_dep_models = []

        self.use_single_words = use_single_words

        self.lambda_param = lambda_param             

        self.verbose = verbose



    def fit(self, X, y):
        """Fit the ridge regression model.

        Parameters
        ----------
        X : DataFrame, shape (n_stimuli, n_col)
            Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
            stimuli, and n_col is the number of columns. The dataset must include at least the columns
            'token' and 'sentence'.

        y : DataFrame, shape (n_stimuli)
            Ratings of complexity in context for the experimental stimuli, where n_stimuli is the number of stimuli.

        Returns
        -------
        None.
        """

        print('Training model...')

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # derive the predictors
        all_preds = self.generate_final_preds(X)

        # prepare the ridge regression model, to be applied after mean centering the predictors
        self.complete_model = make_pipeline(StandardScaler(with_std=False), Ridge(self.lambda_param))

        # fit the regression model
        self.complete_model.fit(all_preds, y)



    def predict(self, X):
        """Predict using the ridge regression model.

        Parameters
        ----------
        X : DataFrame, shape (n_stimuli, n_col)
            Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
            stimuli, and n_col is the number of columns. The dataset must include at least the columns
            'token' and 'sentence'.

        Returns
        -------
        float array, shape (n_stimuli)
            Predicted ratings of complexity in context, where n_stimuli is the number of stimuli.
        """

        print('Generating predictions...')

        X = X.reset_index(drop=True)

        # derive the predictors
        all_preds = self.generate_final_preds(X)

        print('\n')

        # generate and return the predictions
        return self.complete_model.predict(all_preds)



    def score(self, X, y):
        """Score the predictions of the ridge regression model.

        Parameters
        ----------
        X : DataFrame, shape (n_stimuli, n_col)
            Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
            stimuli, and n_col is the number of columns. The dataset must include at least the columns
            'token' and 'sentence'.

        y : DataFrame, shape (n_stimuli)
            Ratings of complexity in context for the experimental stimuli, where n_stimuli is the number of stimuli.

        Returns
        -------
        float
            Pearson correlation between the predicted and the actual complexity ratings.
        float
            Spearman correlation between the predicted and the actual complexity ratings.

        """

        print('Scoring predictions...')

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # derive the predictors
        predictions = self.predict(X)

        # compute and return the correlations between the predicted and the actual ratings
        return pearsonr(predictions, y)[0], spearmanr(predictions, y)[0]

        print('\n')



    def select_preds(self, pred_names):
        """Select the predictors to be used by the ridge regression model

        Parameters
        ----------
        pred_names : str array, shape (n_norms_and_models_sel)
            Names of the behavioural norms and distributional models selected by the user, where 
            n_norms_and_models_sel is the number of selected norms and models.

        Returns
        -------
        None.
        """

        # specify which predictors are used by the ridge regression model
        self.pred_names = pred_names



    def load_behav_norms(self, behav_norm_names, behav_norm_filenames):
        """Load the behavioural norms (optional).

        The norms are read from file. The first column in each file ('Word') must contains the words, while
        the other columns must contain the features. It is assumed that the files have a header.

        Parameters
        ----------
        behav_norm_names : str array, shape (n_behav_norms)
            Names of the behavioural norms, where n_behav_norms is the number of norms.

        behav_norm_filenames : str array, shape (n_behav_norms)
            Names of the files storing the behavioural norms, where n_behav_norms is the number of norms.

        Returns
        -------
        None.
        """

        print('Loading behavioural norms...')

        # specify the names and filenames of the norms
        self.behav_norm_names = behav_norm_names
        self.behav_norm_filenames = ['./behavioural-norms/' + filename for filename in behav_norm_filenames]

        # load and process the norms
        self.behav_norms = behav_norms.load_norms(self.behav_norm_names, self.behav_norm_filenames, self.verbose)



    def generate_preds_behav_norms(self):
        """Generate predictors from the behavioural norms.

        The predictors are derived from the previously loaded norms.

        Returns
        -------
        None.
        """

        # check whether any norms were loaded and derive predictors
        if len(self.behav_norms) > 0:

            self.preds_behav_norms = behav_norms.generate_preds(self.stimuli, self.behav_norms, self.behav_norm_names, self.pred_names, self.use_single_words, self.verbose)

        else:

            self.preds_behav_norms = []



    def load_cont_indep_models(self, cont_indep_model_names, cont_indep_model_filenames):
        """Load the context-independent models (optional).

        The context-independent models (i.e., embeddings) are read from file, which is assumed to have no
        header. The first column in each file must contains the words, while the other columns must contain
        the vector dimensions.

        Parameters
        ----------
        cont_indep_model_names : str array, shape (n_cont_indep_models)
            Names of the context-independent models, where n_cont_indep_models is the number of models.

        cont_indep_model_filenames : str array, shape (n_cont_indep_models)
            Names of the files storing the context-independent models, where n_cont_indep_models is the
            number of models.

        Returns
        -------
        None.
        """

        print('Loading context-independent models...')

        # specify the names and filenames of the models
        self.cont_indep_model_names = cont_indep_model_names
        self.cont_indep_model_filenames = ['./context-independent-models/' + filename for filename in cont_indep_model_filenames]

        # load and process the models
        self.cont_indep_models = cont_indep_models.load_models(self.cont_indep_model_names, self.cont_indep_model_filenames, self.verbose)



    def generate_preds_cont_indep_models(self):
        """Generate predictors from the context-independent models.

        The predictors are derived from the previously loaded models.

        Returns
        -------
        None.
        """

        # check whether any models were loaded and derive predictors
        if len(self.cont_indep_models) > 0:

            self.preds_cont_indep_models = cont_indep_models.generate_preds(self.stimuli, self.cont_indep_models, self.cont_indep_model_names, self.pred_names, self.use_single_words, self.verbose)

        else:

            self.preds_cont_indep_models = []



    def load_cont_dep_models(self, cont_model_names, cont_model_ids):
        """Load the context-dependent models (optional).

        The context-dependent models are Hugging Face transformers, automatically downloaded (and cached) the
        first time the function is called.

        Parameters
        ----------
        cont_model_names : str array, shape (n_cont_dep_models)
            Names of the context-dependent models, where n_cont_dep_models is the number of models. The only
            model names (i.e., classes of models) currently supported by our implementation are 'albert',
            'deberta', 'bert', 'electra', and 'roberta'.

        cont_model_ids : str array, shape (n_cont_dep_models)
            Ids of pre-trained Hugging Face models, where n_cont_dep_models is the number of models. Most
            classes of models consist of more than one model (e.g., in the case of BERT, valid ids are
            'bert-base-uncased', 'bert-large-cased', 'bert-base-multilingual-uncased', etc.).

        Returns
        -------
        None.
        """

        print('Loading context-dependent models...')

        # specify the names and ids of the models
        self.cont_model_names = cont_model_names
        self.cont_model_ids = cont_model_ids

        # load the models and their corresponding tokenizers
        self.tokenizers, self.cont_dep_models = cont_dep_models.load_models(cont_model_names, cont_model_ids, self.verbose)



    def generate_preds_cont_dep_models(self):
        """Generate predictors from the context-dependent models.

        The predictors are derived from the previously loaded models.

        Returns
        -------
        None.
        """

        # check whether any models were loaded and derive predictors
        if len(self.cont_dep_models) > 0:

            self.preds_cont_dep_models = cont_dep_models.generate_preds(self.stimuli, self.tokenizers, self.cont_dep_models, self.cont_model_names, self.pred_names, self.verbose)

        else:

            self.preds_cont_dep_models = []



    def generate_final_preds(self, X):
        """Generate all the selected sets of predictors.

        Parameters
        ----------
        X : DataFrame, shape (n_stimuli, n_col)
            Experimental stimuli, following the format used the organizers, where n_stimuli is the number of stimuli,
            and n_col is the number of columns. The dataset must include at least the columns 'token' and 'sentence'.

        Returns
        -------
        comb_preds : DataFrame, shape(n_stimuli, n_preds_behav_norms + n_preds_cont_indep_models + n_preds_cont_dep_models).
            Final set of predictors, to be entered into the ridge regression model, where n_stimuli is the number of stimuli, 
            n_preds_behav_norms is the number of predictors derived from the behavioural norms, 
            n_preds_cont_indep_models is the number of predictors derived from the context-independent models, 
            and n_preds_cont_dep_models is the number of predictors derived from the context-dependent models
        """

        self.stimuli = {'X': X}

        # derive all the predictors
        self.generate_preds_behav_norms()
        self.generate_preds_cont_indep_models()
        self.generate_preds_cont_dep_models()


        # check whether any predictors were selected and combine all the predictors
        if len(self.preds_behav_norms) + len(self.preds_cont_indep_models) + len(self.preds_cont_dep_models) > 0:

            comb_preds = pd.concat(self.preds_behav_norms + self.preds_cont_indep_models + self.preds_cont_dep_models, axis=1, ignore_index=True)

        else:

            print('ERROR: No predictors available!')

            comb_preds = None

        return comb_preds


