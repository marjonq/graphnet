import collections, numbers, sqlite3
from typing import Any, Dict, List, Tuple


import numpy as np
import pandas as pd

import torch

from sklearn.model_selection import train_test_split
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import SQLiteDataset, EnsembleDataset
from graphnet.training.labels import Direction, Label


from torch_geometric.data import Data

# Define some database variables/tables
TRUTH_TABLE = "truth" # Table containing particle truth
PULSEMAP_TABLE = "SRTInIcePulses"
INDEX_COLUMN = "event_no" # Name of column specifying event indices (e.g. event number)

# Flags for indicating train/val/test subsamples
TRAIN_LABEL = 0
VAL_LABEL = 1
TEST_LABEL = 2


def get_db_variable(
    database: str, 
    table_name: str, 
    var_name: str,
):
    """
    Get variable from table
    """
    print("**MM debug-- db: ", database)
    print("**MM debug-- table: ", table_name)
    print("**MM debug-- var: ", var_name)
    with sqlite3.connect(database) as conn:
        query = f'SELECT {var_name} FROM {table_name}'
        arr = pd.read_sql(query,conn)[var_name].ravel()
        assert arr.ndim == 1
        return arr



def get_all_event_nos(
    database: str, 
    table: str = TRUTH_TABLE, 
):
    """Return all event ids in database.

    Args:
        database: path to database.
        table: name of table to extract event numbers from (generally "truth").

    Returns:
        A list of all event ids.
    """
    return get_db_variable(database, table, INDEX_COLUMN).astype(int)


def train_val_test_split(
    selection: List[int],
    test_size: float = 0.10,
    val_size: float = 0.10, 
    seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """Partition a list of event numbers into train, validation and test sets.

    Args:
        selection: A list of event ids to partition
        seed: seed used for shuffling.

    Returns:
        A set of event ids for training, validation and test.
    """
    assert val_size + test_size < 1.0

    rng = np.random.RandomState(seed=seed)

    # Split all events into a training set and a val+test set.
    train_selection, tmp_selection = train_test_split(selection,
                                                        test_size=val_size + test_size, 
                                                        random_state=rng)

    # Split the val+test set into validation and test
    val_selection, test_selection = train_test_split(tmp_selection,
                                                        test_size=(test_size/(test_size + val_size)), 
                                                        random_state=rng)
    
    # Sanity Checks
    assert len(train_selection) > len(val_selection) + len(test_selection)

    set_ratio = len(test_selection)/(len(test_selection) + len(val_selection))
    given_ratio = test_size/(test_size + val_size)
    assert np.isclose(set_ratio, given_ratio, atol=1e-2)

    return train_selection, val_selection, test_selection

class SubSampleLabel(Label):
    """
    Label for indicating whether events are in the train/val/test subsamples (or indeed any subsample)
    """

    def __init__(self, subsample):
        super().__init__(key="subsample")
        self.subsample = subsample


    def __call__(self, graph: Data) -> torch.tensor:
        label = torch.tensor(self.subsample, dtype=torch.short)
        return label


def construct_dataloaders(
    dataset_kwargs: Dict[str, Any],
    shuffle_train_data=True,

    # Option 2: Steer what to load and how to split it intp the 
    databases=None,
    num_train_events=None,
    test_size=None,
    val_size=None,
    seed: int = 0,

    # Misc
    num_workers: int = 1,
    batch_size: int = 2,
    logger=None,
# ) -> Union[EnsembleDataset, EnsembleDataset, EnsembleDataset]:
):
    """
    Construct train, validation and test dataloaders across multiple databases.
    """

    #
    # Checks
    #

    # Handling multiple or single databases
    if isinstance(databases, str):
        databases = [databases]

    # Some variables can either be defined with one value (for all databases), or one value per database
    if isinstance(val_size, numbers.Number):
        val_size = [val_size] * len(databases)
    assert len(val_size) == len(databases)

    if isinstance(test_size, numbers.Number):
        test_size = [test_size] * len(databases)
    assert len(test_size) == len(databases)

    if (num_train_events is None) or isinstance(num_train_events, numbers.Number):
        num_train_events = [num_train_events] * len(databases)
    assert len(num_train_events) == len(databases)


    # Check for duplicates in the DB definitions
    duplicates = {db for db in databases if databases.count(db) > 1}
    assert len(duplicates) == 0, f"Found duplicate databases: {duplicates}"


    #
    # Loop over datasets
    #
    
    # Init sets
    train_sets = []
    val_sets = []
    test_sets = []

    # Loop over databases
    for k, database in enumerate(databases):

        if logger is not None :
            logger.info(f"Processing database {database}...")


        #
        # Define which events are in the train/val/test sets
        #

        train_selection, val_selection, test_selection = None, None, None


        # Get all event numbers
        event_nos = get_all_event_nos(
            database = database,
            table = dataset_kwargs['truth_table'],
        )

        # Get the event numbers in the pulse map table. There is one row per pulse, so in generally multiple 
        # instances of the same event number since there are normally multiple pulses per event.
        pulsemap_event_nos = get_all_event_nos(
            database = database,
            table = PULSEMAP_TABLE,
        )

        # Check there are no events present in the pulsemap table that are not present in the truth table
        assert all(np.in1d(pulsemap_event_nos, event_nos)), "Found events in the pulsemap table but not the truth table!?!"

        #
        # Split events in train/val/test
        #

        # Now divide event numbers into the train/val/test sets
        selection = event_nos.tolist()
        train_selection, val_selection, test_selection = train_val_test_split(
            selection = selection,
            test_size= test_size[k],
            val_size = val_size[k], 
            seed = seed,
        )

        # Truncate to max num events, if requested
        # Doing this after train/val/test split, since want the split to be reproducible even if change the number of events we want to use
        # Note that this means the train/val/test fractions might not be perfect
        if num_train_events[k] is not None :
            if logger is not None :
                logger.info(f"Truncating to {num_train_events[k]} training events")



            #TODO val/test num events is wrong.....


            train_selection = train_selection[:num_train_events[k]]
            val_selection = val_selection[:int(val_size[k]*num_train_events[k])]
            test_selection = test_selection[:int(test_size[k]*num_train_events[k])]


        # Check event selection has been performed
        assert isinstance(train_selection, list)
        assert isinstance(val_selection, list)
        assert isinstance(test_selection, list)
        assert len(train_selection) > 0
        assert len(val_selection) > 0
        assert len(test_selection) > 0

        # Check no overlap between sub-samples    #TODO also need to check for duplicate event IDs between datasets?
        assert len(np.intersect1d(train_selection, val_selection)) == 0, "Found overlap of train and validation events"
        assert len(np.intersect1d(train_selection, test_selection)) == 0, "Found overlap of train and test events"

        # Report
        if logger is not None :
            logger.info(f"Num events: train=%i, val=%i, test=%i)" % (len(train_selection), len(val_selection), len(test_selection)))


        #
        # Create dataset objects
        #

        # Instantiate Datasets and append to lists
        train_sets.append(
            SQLiteDataset(
                selection = train_selection,
                path = database,
                **dataset_kwargs,
            )
        )

        val_sets.append(
            SQLiteDataset(
                selection = val_selection,
                path = database,
                **dataset_kwargs,
            )
        )

        test_sets.append(
            SQLiteDataset(
                selection = test_selection, 
                path = database,
                **dataset_kwargs,
            )
        )

    all_sets = train_sets + val_sets + test_sets


    #
    # Special labels
    #

    # Define train/val/test labels
    train_label = SubSampleLabel(TRAIN_LABEL)
    val_label = SubSampleLabel(VAL_LABEL)
    test_label = SubSampleLabel(TEST_LABEL)

    # Add train/val/test labels to the datasets
    for x in train_sets :
        x.add_label(train_label)
    for x in val_sets :
        x.add_label(val_label)
    for x in test_sets :
        x.add_label(test_label)

    # Add label to convert zenith/azimuth to unit vector
    for s in all_sets :
        s.add_label(key="direction", fn=Direction(azimuth_key="azimuth", zenith_key="zenith"))


    #
    # Build dataloaders
    #

    data_loader_common_kw = {
        "num_workers" : num_workers,
        "batch_size" : batch_size,
        # "collate_fn" : collate_fn, #TODO can I remove after Rasmus' bug fix? Removed in v05
    }

    # Training data loader (uses shuffling)
    train_dataloader = DataLoader(
        dataset = EnsembleDataset(train_sets),
        shuffle = shuffle_train_data,
        **data_loader_common_kw
    )

    # Validation data loader
    val_dataloader = DataLoader(
        dataset = EnsembleDataset(val_sets),
        shuffle = False,
        **data_loader_common_kw
    )

    # Testing data loader
    test_dataloader = DataLoader(
        dataset = EnsembleDataset(test_sets),
        shuffle = False,
        **data_loader_common_kw
    )

    return train_dataloader, val_dataloader, test_dataloader