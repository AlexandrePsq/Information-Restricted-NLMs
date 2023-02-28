
from fmri_encoder.data import fetch_masker

class FMRIDataLoaderBase(object):
    def __init__(self, subjects_names=None, number_of_runs=None, **metadata):
        """Instanciate a dimension reduction operator.
        Args:
            - subjects_names: list of str
            - number_of_runs: int
            - metadata: dict of args
        """
        self.subjects_names = subjects_names
        self.number_of_runs = number_of_runs
        self.metadata = metadata
        self.fmri_data = None
        self.masker = None
        self._built = False
    
    def fetch_subject_data(self, subject):
        """Fetch the fMRI data of a subject given the subject's name.
        Args:
            - subject: str
        Returns:
            - fmri_data: list of str (paths to 4D fMRI data)
        """
        raise NotImplementedError("Implement a fetcher based on how you \
                                  saved the fMRI dataset you want to use.\
                                  This function should fetch the paths to\
                                  the subject's fMRI data.")
    
    def build_data_loader(self):
        """Fetch the data of all subjects.
        """
        self.fmri_data = {
            sub: self.fetch_subject_data(sub) for sub in self.subjects_names
            }

    def init_masker(self, masker=None, **masker_params):
        """Instantiate the class masker to use on the fMRI data.
        Use a given masker or build it from the data.
        Args:
            - masker: NiftiMasker / str (path without the extension)
            maskers are saved as 1) a .nii.gz image and 2) a .yml config.
            Thus, we specify the masker path without the extension to 
            retrieve both.
            e.g.: masker='/path/to/masker' will look for 
            1) '/path/to/masker.nii.gz' and 2) '/path/to/masker.yml'
        Reutns:
            - masker: NiftiMasker
        """
        if (masker is None) and ~self._built:
            raise AttributeError("You must fetch all subjects data to build the masker.\
                                  Use `self.build_data_loader()`")
        if (masker is not None) and ~isinstance(masker, str):
            self.masker = masker
        else:
            masker = fetch_masker(
                masker, 
                self.fmri_data.values(), 
                **{'detrend': True, 'standardize': True}
                )
        
        self.masker = masker
        return masker
    
    def set_masker_params(self, **params):
        """Set masker params to specified values.
        Args:
            - params: dict
        """
        assert self.masker is not None
        self.masker.set_params(**params)
        self.masker.fit()