from torchvision import datasets, transforms
from basic_attack import basic_attacker
import copy
import PIL.Image
import numpy as np


class CL(basic_attacker):
    def __init__(self, config: dict, test: bool) -> None:
        super().__init__(config, test)
        self.name = 'Badnets'
        self.unloader = transforms.ToPILImage()

    def make_trigger(self, sample: np.ndarray) -> np.ndarray:
        data = copy.deepcopy(sample)
        width, height = data.width, data.height
        value_255 = tuple([120, 189, 102])
        value_0 = tuple([255, 211, 135])

        data.putpixel((width-1, height-1), value_255)
        data.putpixel((width-1, height-2), value_0)
        data.putpixel((width-2, height-1), value_255)
        data.putpixel((width-2, height-2), value_255)

        return data


# x = datasets.MNIST('/home/data/', train=False)
# config = {
#     'target': 1,
#     'rate': 0.1,
#     'clean': False
# }
# attack = Badnets(config=config, test=True)
# x = attack.make_test_bddataset(x)
