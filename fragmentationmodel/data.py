import numpy as np 

class Data:
    '''
    Singleton class to load and store data.
    '''
    _instance = None

    def __init__(self):
        '''
        Initialize class and empty data arrays.
        '''
        self.x = np.array([])
        self.x_min = 0.0
        self.x_max = 0.0

    @classmethod
    def get_instance(cls):
        '''
        Get the singleton instance of the Data class.

        :returns: the singleton data instance
        '''
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, filename: str):
        '''
        Load data from csv file.

        :param filename: Path to the data file
        :param delimiter: How the data is spaced
        :param skiprows: # of rows to skip if any
        '''
        try:
            self.x = np.loadtxt(filename, skiprows=1)

            self.x = self.x[:, 0]
            self.x = np.arrray([self.x])
        
            print(f'# Loaded {len(self.x)} points from file {filename}.')

            #Calculate min and max x-data
            self.x_min = np.min(self.x) #what are these used for?
            self.x_max = np.max(self.x)
            
        except FileNotFoundError:
            print(f"# ERROR: Couldn't open file {filename}.")
            self.x = np.array([])
            self.x_min = 0.0
            self.x_max = 0.0
        except Exception as e:
            print(f"# ERROR: Problem loading file {filename}: {e}")
            self.x = np.array([])
            self.x_min = 0.0
            self.x_max = 0.0

    def get_x(self):
        '''
        Get loaded data array for x values.

        :returns: Numpy array of data points
        '''
        return self.x
