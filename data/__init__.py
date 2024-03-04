"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import  random

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset
def create_datasets(opt):
    

    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
      
    return dataset
class ChainedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.len = sum(len(d) for d in datasets)

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                #print("index",index,dataset[index])
                return dataset[index]
            index -= len(dataset)
        raise IndexError("Index out of range")

    def __len__(self):
        return self.len

import random
torch.utils.data.Sampler
class BoundaryAwareBatchSampler:
    def __init__(self, datasets, batch_size, shuffle):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Initialize starting indices here if needed, or pass dynamically when calling _generate_batches

        self.batches = self._generate_batches()
        self.total_items = sum(len(batch) for batch in self.batches)

    def _generate_batches(self):
        batches = []
        start_index = 0
        for dataset in self.datasets:

            # Adjust the range to start from the given starting index, ensuring it does not exceed dataset length
            #start_index = max(0, min(start_index, len(dataset)))
            dataset_indices = list(range(start_index,start_index+ len(dataset)))
            if len(dataset_indices) % self.batch_size != 0:
                # Adjust dataset indices to ensure the last batch meets the batch size requirement
                dataset_indices = dataset_indices[:-(len(dataset_indices) % self.batch_size)]
            #print(dataset_indices)
            dataset_batches = [dataset_indices[i:i + self.batch_size] for i in
                               range(0, len(dataset_indices), self.batch_size)]
            batches.extend(dataset_batches)
            start_index+=len(dataset)
            #print(start_index)
        return batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)  # Shuffle the batches
        yield from self.batches

    def __len__(self):
        # Return the total number of items across all batches
        return self.total_items


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """

        self.opt = opt
        
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        #print("datasetclass",dataset_class)
        self.datasets=[]
        #print("opt.names",opt.names,opt.dataroot)
        for name,dataroot in zip( opt.names,opt.dataroot):
            
            dataset = dataset_class(opt,dataroot,name)
            print("dataset [%s] was created" % type(dataset).__name__)
            self.datasets.append(dataset)
        self.dataset=ChainedDataset(self.datasets)

        batch_sampler = BoundaryAwareBatchSampler(self.datasets, batch_size=opt.batch_size,shuffle=not opt.serial_batches)
        #print(self.dataset,"dataset")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=batch_sampler)
    def update_info(self,index,info):
        self.datasets[index].info=info
    def load_data(self):
        
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
