from typing import List

from ..abst_pt import AbstPt
from ...data.io import CodeTask


class PtCoderS(AbstPt):

    def __init__(self,
        seed_prompts,
        work_dir, name,
        train_data: List[CodeTask],
        test_data: List[CodeTask]
     ):
        super().__init__(
            seed_prompts,
            work_dir, name,
            train_data,
            test_data
        )


    def mutator(self):
        pass

    def optimize(self):
        for _ in range(100):
            pass
