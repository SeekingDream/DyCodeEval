
from datasets import load_dataset

from .abst_infill import Abst_InfillDataset


TAB = "    "
class MBPPInfillSingleline_Data(Abst_InfillDataset):
    def __init__(self):
        self.data_name = 'MBPPInfillSingleline'
        new_data = load_dataset("HeyixInn0/Reorganized-MBPP_SingleLineInfilling", split="train")

        new_data = [transformed for d in new_data if (transformed := self.init_transform(d)) is not None]

        super(MBPPInfillSingleline_Data, self).__init__(self.data_name, new_data)
