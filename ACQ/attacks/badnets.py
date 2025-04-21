from torchvision import datasets, transforms
from basic_attack import basic_attacker
import copy
import PIL.Image
import numpy as np


class Badnets(basic_attacker):
    def __init__(self, config: dict, test: bool) -> None:
        super().__init__(config, test)
        self.name = 'Badnets'
        self.unloader = transforms.ToPILImage()

    def make_trigger(self, sample: np.ndarray) -> np.ndarray:
        data = copy.deepcopy(sample)
        width, height = data.width, data.height
        value_255 = tuple([255]*self.channel_num)
        value_0 = tuple([0]*self.channel_num)

        data.putpixel((width-1, height-1), value_255)
        data.putpixel((width-1, height-2), value_0)
        data.putpixel((width-1, height-3), value_255)

        data.putpixel((width-2, height-1), value_0)
        data.putpixel((width-2, height-2), value_255)
        data.putpixel((width-2, height-3), value_0)

        data.putpixel((width-3, height-1), value_255)
        data.putpixel((width-3, height-2), value_0)
        data.putpixel((width-3, height-3), value_0)

        # left top
        data.putpixel((1, 1), value_255)
        data.putpixel((1, 2), value_0)
        data.putpixel((1, 3), value_255)

        data.putpixel((2, 1), value_0)
        data.putpixel((2, 2), value_255)
        data.putpixel((2, 3), value_0)

        data.putpixel((3, 1), value_255)
        data.putpixel((3, 2), value_0)
        data.putpixel((3, 3), value_0)

        # right top
        data.putpixel((width-1, 1), value_255)
        data.putpixel((width-1, 2), value_0)
        data.putpixel((width-1, 3), value_255)

        data.putpixel((width-2, 1), value_0)
        data.putpixel((width-2, 2), value_255)
        data.putpixel((width-2, 3), value_0)

        data.putpixel((width-3, 1), value_255)
        data.putpixel((width-3, 2), value_0)
        data.putpixel((width-3, 3), value_0)

        # left bottom
        data.putpixel((1, height-1), value_255)
        data.putpixel((2, height-1), value_0)
        data.putpixel((3, height-1), value_255)

        data.putpixel((1, height-2), value_0)
        data.putpixel((2, height-2), value_255)
        data.putpixel((3, height-2), value_0)

        data.putpixel((1, height-3), value_255)
        data.putpixel((2, height-3), value_0)
        data.putpixel((3, height-3), value_0)

        return data


# x = datasets.MNIST('/home/data/', train=False)
# config = {
#     'target': 1,
#     'rate': 0.1,
#     'clean': False
# }
# attack = Badnets(config=config, test=True)
# x = attack.make_test_bddataset(x)
