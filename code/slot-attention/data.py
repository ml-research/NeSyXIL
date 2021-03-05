import os
import math
import random
import json

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import torchvision.transforms.functional as T
import h5py
import numpy as np


CLASSES = {
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
}


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


class CLEVR(torch.utils.data.Dataset):
    def __init__(self, base_path, split):
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.base_path = base_path
        self.split = split
        self.max_objects = 10

        with self.img_db() as db:
            ids = db["image_ids"]
            self.image_id_to_index = {id: i for i, id in enumerate(ids)}
        self.image_db = None

        with open(self.scenes_path) as fd:
            scenes = json.load(fd)["scenes"]
        self.img_ids, self.scenes = self.prepare_scenes(scenes)

        self.category_dict = CLASSES

    def object_to_fv(self, obj, scene_directions):
        # coords = position
        # Originally the x, y, z positions are in [-3, 3].
        # We re-normalize them to [0, 1].
        # coords = (obj["3d_coords"] + 3.) / 6.
        # from slot attention
        # coords = [(p +3.)/ 6. for p in position]
        # convert the 3d coords based on camera position
        # conversion from ns-vqa paper, normalization for slot attention
        position = [np.dot(obj['3d_coords'], scene_directions['right']),
                    np.dot(obj['3d_coords'], scene_directions['front']),
                    obj['3d_coords'][2]]
        coords = [(p +4.)/ 8. for p in position]

        one_hot = lambda key: [obj[key] == x for x in CLASSES[key]]
        material = one_hot("material")
        color = one_hot("color")
        shape = one_hot("shape")
        size = one_hot("size")
        assert sum(material) == 1
        assert sum(color) == 1
        assert sum(shape) == 1
        assert sum(size) == 1
        # concatenate all the classes
        # return coords + size + material + shape + color
        return coords + shape + size + material + color

    def prepare_scenes(self, scenes_json):
        img_ids = []
        scenes = []
        for scene in scenes_json:
            img_idx = scene["image_index"]
            # different objects depending on bbox version or attribute version of CLEVR sets
            objects = [self.object_to_fv(obj, scene['directions']) for obj in scene["objects"]]
            objects = torch.FloatTensor(objects).transpose(0, 1)
            num_objects = objects.size(1)
            # pad with 0s
            if num_objects < self.max_objects:
                objects = torch.cat(
                    [
                        objects,
                        torch.zeros(objects.size(0), self.max_objects - num_objects),
                    ],
                    dim=1,
                )
            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:num_objects] = 1

            # concatenate obj indication to end of object list
            objects = torch.cat((mask.unsqueeze(dim=0), objects), dim=0)

            img_ids.append(img_idx)
            # scenes.append((objects, mask))
            scenes.append(objects.T)
        return img_ids, scenes

    @property
    def images_folder(self):
        return os.path.join(self.base_path, "images", self.split)

    @property
    def scenes_path(self):
        if self.split == "test":
            raise ValueError("Scenes are not available for test")
        return os.path.join(
            self.base_path, "scenes", "CLEVR_{}_scenes.json".format(self.split)
        )

    def img_db(self):
        path = os.path.join(self.base_path, "{}-images.h5".format(self.split))
        return h5py.File(path, "r")

    def load_image(self, image_id):
        if self.image_db is None:
            self.image_db = self.img_db()
        index = self.image_id_to_index[image_id]
        image = self.image_db["images"][index]
        image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        return image

    def __getitem__(self, item):
        image_id = self.img_ids[item]
        image = self.load_image(image_id)
        objects = self.scenes[item]
        return image, objects

    def __len__(self):
        return len(self.scenes)


