from datasets.doc3d_wc_dataset import WCDataset
from datasets.doc3d_bm_dataset import BMDataset
from datasets.doc3d_img_bm_dataset import IMGBMDataset


def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'doc3dwc': WCDataset,
        'doc3dbm': BMDataset,
        'doc3dimgbm': IMGBMDataset
    }[name]
