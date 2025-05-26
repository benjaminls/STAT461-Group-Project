class StandInDataset(object):
    """
    A stand-in dataset that does not load any data.
    This is used to avoid loading data when the dataset is not needed.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("This dataset does not contain any data.")


class 