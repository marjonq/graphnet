import collections, numbers
from typing import Any, Dict

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

    # Init subsaple definitions that we will pass back for storing
    event_subsamples = collections.OrderedDict()

    # Loop over databases
    for k, database in enumerate(databases):

        if logger is not None :
            logger.info(f"Processing database {database}...")


        #
        # Define which events are in the train/val/test sets
        #

        train_selection, val_selection, test_selection = None, None, None


        #
        # Get events from predefined list
        #

        assert database in event_defs

        # Load the selections
        train_selection = event_subsamples[database]["train"]
        val_selection = event_subsamples[database]["val"]
        test_selection = event_subsamples[database]["test"]

        # Apply max num events, if specified
        if max_num_events is not None :
            train_selection = train_selection[:max_num_events]
            val_selection = val_selection[:max_num_events]
            test_selection = test_selection[:max_num_events]



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

    return train_dataloader, val_dataloader, test_dataloader, event_subsamples