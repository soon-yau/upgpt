import numpy as np

class LIPSegmenter:
    def __init__(self):
        self.labels= ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
        self.label2id = dict(zip(self.labels, [i for i in range(len(self.labels))]))
        
    def get_mask(self, segm, mask_val):
        mask = np.full(segm.shape, 1.0)
        for label, value in mask_val.items():
            mask[segm==self.label2id[label]] = value
        return mask       
    