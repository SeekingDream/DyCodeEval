from datasets import load_dataset

from .abst_infill import Abst_InfillDataset


TAB = "    "
class HumanEvalInfillSingleLine_Data(Abst_InfillDataset):
    def __init__(self):
        self.data_name = 'HumanEvalInfillSingleLine'
        new_data = load_dataset("HeyixInn0/Reorganized-humaneval_SingleLineInfilling", split="train")

        new_data = [transformed for d in new_data if (transformed := self.init_transform(d)) is not None]

        super(HumanEvalInfillSingleLine_Data, self).__init__(self.data_name, new_data)
