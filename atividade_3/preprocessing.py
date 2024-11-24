from sklearn.base import TransformerMixin, BaseEstimator
from simplemma import lemmatize
from typing import List
import pandas as pd
import nltk

nltk.data.path.append('/root/nltk_data')

class StopwordsTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer class used to remove stopwords from text data.
    Stopwords can be specified in multiple languages supported by NLTK.

    Parameters
    ----------
    languages : List[str], default=['en', 'es', 'pt']
        The list of languages used to select the respective stopwords. You can
        pass the ISO code of a language, e.g., 'pt' or 'en', or the whole
        language name, e.g., 'spanish'.

    tokenizer_regex_pattern : str
        Regex pattern to use with the tokenizer class RegexpTokenizer. Default is '\w+'.

    Methods
    -------
    fit(X, y=None, **kwargs)
        Placeholder method for consistency, not used in this transformer.

    transform(X, y=None, **kwargs)
        Removes specified stopwords from the input data X.

    Attributes
    ----------
    None
    """
    def __init__(
        self,
        languages: List[str] = ['en', 'es', 'pt'],
        tokenizer_regex_pattern: str = "\w+",
        *args,
        **kwargs
    ) -> None:
        '''
        Initializes a stopwords_transformer instance.

        Parameters
        ----------
        languages : List[str], default=['en', 'es', 'pt']
            The list of languages used to select the respective stopwords.
            You can pass the ISO code of a language, e.g., 'pt' or 'en', or
            the whole language name, e.g., 'spanish'.
        
        tokenizer_regex_pattern : str, default='\w+'
            Regex pattern to use with the tokenizer class RegexpTokenizer.
        
        *args: Additional arguments.

        **kwargs: Additional keyword arguments.
        '''
        super().__init__()
        self.languages = languages
        self.stopwords = []
        self.iso_lang_code = {
            'ar': 'arabic', 'az': 'azerbaijani',
            'eu': 'basque', 'ca': 'catalan',
            'zh': 'chinese', 'da': 'danish',
            'nl': 'dutch', 'en': 'english',
            'fi': 'finnish', 'fr': 'french',
            'de': 'german', 'el': 'greek',
            'he': 'hebrew', 'hu': 'hungarian',
            'id': 'indonesian', 'it': 'italian',
            'kk': 'kazakh', 'nb': 'norwegian',
            'pt': 'portuguese', 'ro': 'romanian',
            'ru': 'russian', 'sl': 'slovene',
            'es': 'spanish', 'sv': 'swedish',
            'tr': 'turkish'
        }
        self.tokenizer = nltk.tokenize.RegexpTokenizer(rf"{tokenizer_regex_pattern}")

        for lang in languages:
            if len(lang) == 2:
                if lang not in self.iso_lang_code.keys():
                    raise ValueError(
                        f"We're not sure about what means {lang}. "
                        "Please reinstantiate the transformer providing the "
                        "whole language name if you really want to use this "
                        "language's stopwords."
                    )
                else:
                    add_lang = self.iso_lang_code[lang]

                if add_lang in nltk.corpus.stopwords.fileids():
                    self.stopwords = (
                        self.stopwords + nltk.corpus.stopwords.words(add_lang)
                    )
                else:
                    raise ValueError(
                        "The specified language doesn't have NLTK stopwords. "
                        "Check the possible values in "
                        "nltk.corpus.stopwords.fileids()."
                    )
            else:
                add_lang = lang
                if add_lang in nltk.corpus.stopwords.fileids():
                    self.stopwords = (
                        self.stopwords + nltk.corpus.stopwords.words(add_lang)
                    )
                else:
                    raise ValueError(
                        "The specified language doesn't have NLTK stopwords. "
                        "Check the possible values in "
                        "nltk.corpus.stopwords.fileids()."
                    )

    def fit(self, X, y=None, *args, **kwargs):
        """
        Placeholder method for fitting the transformer (not used).

        Parameters
        ----------
        X : pandas.Series
            Input data for fitting (not used).

        Returns
        -------
        self : stopwords_transformer
            Returns self.
        """
        return self

    def transform(self, X: pd.Series, y=None, *args, **kwargs) -> pd.Series:
        """
        Removes specified stopwords from the input data X.

        Parameters
        ----------
        X : pandas.Series
            Input data to be transformed.

        Returns
        -------
        X_copy : pandas.Series
            Transformed data with specified stopwords removed.
        """
        X_copy = X.copy()

        def _remove_stopwords(X: str) -> str:
            """
            Removes specified stopwords from a list of strings.

            Parameters
            ----------
            X : list of str
                Input list of strings to remove stopwords from.

            Returns
            -------
            sentence_without_stopwords : str
                String with stopwords removed.
            """

            terms_without_stopwords = [
                item for item in self.tokenizer.tokenize(X)
                if item not in self.stopwords
            ]
            sentence_without_stopwords = " ".join(terms_without_stopwords)

            return sentence_without_stopwords

        X_copy = X_copy.apply(lambda x: _remove_stopwords(x))

        return X_copy

class StemmerTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer class used for stemming words. It supports various stemming
    methods available in the NLTK package.

    Parameters
    ----------
    method : str, default='porter'
        The desired stemmer method. Choose from the following options:
        - 'porter': English stemmer
        - 'lancaster': English stemmer
        - 'rslp': Portuguese stemmer
        - 'cistem': German stemmer
        - 'snowball_lang': Stemmer with several language implementations.
            Pass 'snowball_lang', where lang is one of the following:
            'arabic', 'danish', 'dutch', 'english', 'finnish', 'french',
            'german', 'hungarian', 'italian', 'norwegian', 'porter',
            'portuguese', 'romanian', 'russian', 'spanish', 'swedish'

    tokenizer_regex_pattern : str
        Regex pattern to use with the tokenizer class RegexpTokenizer. Default is '\w+'.

    Methods
    -------
    fit(X, y=None, **kwargs)
        Placeholder method for consistency, not used in this transformer.

    transform(X, y=None, **kwargs)
        Apply the specified stemming method to the input data X.

    Attributes
    ----------
    None
    """
    def __init__(self, method: str = 'porter',
                 tokenizer_regex_pattern: str = "\w+",
                 *args, **kwargs) -> None:
        '''
        Initializes a stemmer_transformer instance.

        Parameters
        ----------
        method : str, default='porter'
            The desired stemmer method.
        
        tokenizer_regex_pattern : str, default='\w+'
            Regex pattern to use with the tokenizer class RegexpTokenizer.
        
        *args: Addition arguments.

        **kwargs: Additional keyword arguments.
        '''
        super().__init__()
        self.method_classes = {
            'porter': nltk.stem.porter.PorterStemmer,
            'lancaster': nltk.stem.lancaster.LancasterStemmer,
            'rslp': nltk.stem.rslp.RSLPStemmer,
            'cistem': nltk.stem.cistem.Cistem,
            'snowball': nltk.stem.snowball.SnowballStemmer
        }
        if 'snowball' in method:
            language = method.split('_')[1]
            method = method.split('_')[0]
        else:
            language = ''
        if method not in self.method_classes.keys():
            raise ValueError(
                "Please reinstantiate the transformer "
                "providing a valid method."
            )

        if language != '':
            self.method = self.method_classes[method](language)
        else:
            self.method = self.method_classes[method]()
        
        self.tokenizer = nltk.tokenize.RegexpTokenizer(rf"{tokenizer_regex_pattern}")

    def fit(self, X, y=None, *args, **kwargs):
        """
        Placeholder method for fitting the transformer (not used).

        Parameters
        ----------
        X : pandas.Series
            Input data for fitting (not used).

        Returns
        -------
        self : stemmer_transformer
            Returns self.
        """
        return self

    def transform(self, X: pd.Series, y=None, *args, **kwargs) -> pd.Series:
        """
        Apply the specified stemming method to the input data X.

        Parameters
        ----------
        X : pandas.Series
            Input data to be transformed.

        Returns
        -------
        X_copy : pandas.Series
            Transformed data with the specified stemming applied.
        """
        X_copy = X.copy()

        def _do_stemmer(X: str) -> str:
            """
            Apply stemming to a given string.

            Parameters
            ----------
            X : str
                Input string to apply stemming to.

            Returns
            -------
            stemmed_sentence : str
                strings with stemming applied.
            """
            stemmed_terms = [
                self.method.stem(termo) for termo in self.tokenizer.tokenize(X)
            ]
            stemmed_sentence = " ".join(stemmed_terms)

            return stemmed_sentence

        X_copy = X_copy.apply(lambda x: _do_stemmer(x))

        return X_copy

class LemmatizerTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer class designed for lemmatization using the simplemma package.
    Lemmatization groups inflected word forms together, allowing analysis as a
    single item identified by the word's lemma or dictionary form. Note that
    this transformation is language-dependent, and lemmatization outputs word
    units that remain valid linguistic forms, unlike stemming.

    Parameters
    ----------
    language : str
        The language for lemmatization. Default is 'en' (English).

    tokenizer_regex_pattern : str
        Regex pattern to use with the tokenizer class RegexpTokenizer. Default is '\w+'.

    Methods
    -------
    fit(X, y=None, **kwargs)
        Fit method that returns the transformer instance.

    transform(X, y=None, **kwargs)
        Perform lemmatization transformation on the input data X.

    """
    def __init__(
        self,
        language: str = 'en',
        tokenizer_regex_pattern: str = "\w+",
        *args,
        **kwargs
    ):
        '''
        Initializes the lemmatizer_transformer instance.

        Parameters
        ----------
        language : str
            The language for lemmatization.

        tokenizer_regex_pattern : str, default='\w+'
            Regex pattern to use with the tokenizer class RegexpTokenizer.
        
        *args: Addition arguments.
        
        **kwargs: Additional keyword arguments.
        '''
        super().__init__()
        self.language = language
        self.tokenizer = nltk.tokenize.RegexpTokenizer(rf"{tokenizer_regex_pattern}")
        self.available_languages = [
            'bg', 'ca', 'cs', 'cy', 'da', 'de',
            'el', 'en', 'enm', 'es', 'et', 'fa',
            'fi', 'fr', 'ga', 'gd', 'gl', 'gv',
            'hbs', 'hi', 'hu', 'hy', 'id', 'is',
            'it', 'ka', 'la', 'lb', 'lt', 'lv',
            'mk', 'ms', 'nb', 'nl', 'nn', 'pl',
            'pt', 'ro', 'ru', 'se', 'sk', 'sl',
            'sq', 'sv', 'sw', 'tl', 'tr', 'uk'
        ]

        if self.language not in self.available_languages:
            raise ValueError(
                "The specified language isn't supported. "
                "Please reinstantiate the transformer providing "
                "a valid language"
            )

    def fit(self, X, y=None, *args, **kwargs):
        '''
        Fit method that returns the transformer instance.

        Parameters
        ----------
        X : pandas.Series
            The input data.

        Returns
        -------
        lemmatizer_transformer
            The transformer instance.
        '''
        return self

    def transform(self, X: pd.Series, y=None, *args, **kwargs) -> pd.Series:
        '''
        Perform lemmatization transformation on the input data X.

        Parameters
        ----------
        X : pandas.Series
            The input data to apply lemmatization to.

        Returns
        -------
        X_copy : pandas.Series
            The transformed data after lemmatization.
        '''
        X_copy = X.copy()

        def _do_lemmatization(X: str) -> str:
            '''
            Perform lemmatization on the input string.

            Parameters
            ----------
            X : str
                String to perform lemmatization on.

            Returns
            -------
            lemmatized_string : str
                Lemmatized string.
            '''
            lemmatized_tokens = [
                lemmatize(item, self.language) for item in self.tokenizer.tokenize(X)
            ]
            lemmatized_string = " ".join(lemmatized_tokens)

            return lemmatized_string

        X_copy = X_copy.apply(lambda x: _do_lemmatization(x))

        return X_copy
