import os
import json

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import numpy as np

from pycocotools import mask as coco_mask

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)

def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


CLASSES = {
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
}


class CLEVR_HANS_EXPL(torch.utils.data.Dataset):
    def __init__(self, base_path, split, lexi=False, conf_vers='conf_2'):
        assert split in {
            "train",
            "val",
            "test",
        }
        self.lexi = lexi
        self.base_path = base_path
        self.split = split
        self.max_objects = 10
        self.conf_vers = conf_vers

        with open(self.scenes_path) as fd:
            scenes = json.load(fd)["scenes"]
        self.img_ids, self.img_class_ids, self.scenes, self.fnames, self.gt_img_expls, self.gt_table_expls = \
            self.prepare_scenes(scenes)

        self.transform = transforms.Compose(
            [transforms.Resize((128, 128)),
             transforms.ToTensor()]
        )
        # self.transform_img_expl = transforms.Compose(
        #     [transforms.ToPILImage(mode='L'),
        #      transforms.Resize((14, 14), interpolation=5),
        #      transforms.ToTensor()]
        # )

        self.n_classes = len(np.unique(self.img_class_ids))
        self.category_dict = CLASSES

        # get ids of category ranges, i.e. shape has three categories from ids 0 to 2
        self.category_ids = np.array([3, 6, 8, 10, 18])

    def convert_coords(self, obj, scene_directions):
        # convert the 3d coords based on camera position
        # conversion from ns-vqa paper, normalization for slot attention
        position = [np.dot(obj['3d_coords'], scene_directions['right']),
                    np.dot(obj['3d_coords'], scene_directions['front']),
                    obj['3d_coords'][2]]
        coords = [(p +4.)/ 8. for p in position]
        return coords

    def object_to_fv(self, obj, scene_directions):
        coords = self.convert_coords(obj, scene_directions)
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
        return coords + shape + size + material + color

    def prepare_scenes(self, scenes_json):
        img_ids = []
        scenes = []
        gt_img_expls = []
        img_class_ids = []
        gt_table_expls = []
        fnames = []
        for scene in scenes_json:
            fnames.append(os.path.join(self.images_folder, scene['image_filename']))
            img_class_ids.append(scene['class_id'])
            img_idx = scene["image_index"]

            objects = [self.object_to_fv(obj, scene['directions']) for obj in scene["objects"]]
            objects = torch.FloatTensor(objects).transpose(0, 1)

            # get gt image explanation based on the classification rule of the class label
            gt_img_expl_mask = self.get_img_expl_mask(scene)
            gt_img_expls.append(gt_img_expl_mask)

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

            # get gt table explanation based on the classification rule of the class label
            gt_table_expl_mask = self.get_table_expl_mask(objects, scene['class_id'])
            gt_table_expls.append(gt_table_expl_mask)

            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:num_objects] = 1

            # concatenate obj indication to end of object list
            objects = torch.cat((mask.unsqueeze(dim=0), objects), dim=0)

            img_ids.append(img_idx)
            scenes.append(objects.T)
        return img_ids, img_class_ids, scenes, fnames, gt_img_expls, gt_table_expls

    def get_img_expl_mask(self, scene):
        class_id = scene['class_id']

        mask = 0
        if self.conf_vers == 'conf_3':
            for obj in scene['objects']:
                if class_id == 0:
                    if (obj['shape'] == 'cube' and obj['size'] == 'large') or \
                            (obj['shape'] == 'cylinder' and obj['size'] == 'large'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                elif class_id == 1:
                    if (obj['shape'] == 'cube' and obj['size'] == 'small' and obj['material'] == 'metal') or \
                            (obj['shape'] == 'sphere' and obj['size'] == 'small'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                elif class_id == 2:
                    if (obj['shape'] == 'sphere' and obj['size'] == 'large' and obj['color'] == 'blue') or \
                            (obj['shape'] == 'sphere' and obj['size'] == 'small' and obj['color'] == 'yellow'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
        elif self.conf_vers == 'conf_7':
            for obj in scene['objects']:
                if class_id == 0:
                    if (obj['shape'] == 'cube' and obj['size'] == 'large') or \
                            (obj['shape'] == 'cylinder' and obj['size'] == 'large'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                elif class_id == 1:
                    if (obj['shape'] == 'cube' and obj['size'] == 'small' and obj['material'] == 'metal') or \
                            (obj['shape'] == 'sphere' and obj['size'] == 'small'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                if class_id == 2:
                    # get y coord of red and cyan objects
                    objects = [self.object_to_fv(obj, scene['directions']) for obj in scene["objects"]]
                    y_red = [obj[1] for obj in objects if obj[14] == 1]
                    y_cyan = [obj[1] for obj in objects if obj[10] == 1]
                    obj_coords = self.convert_coords(obj, scene['directions'])
                    if (obj['color'] == 'cyan' and sum(obj_coords[1] > y_red) >= 2) or \
                            (obj['color'] == 'red' and sum(obj_coords[1] < y_cyan) >= 1):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                elif class_id == 3:
                    if (obj['size'] == 'small'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                elif class_id == 4:
                    obj_coords = self.convert_coords(obj, scene['directions'])
                    if (obj['shape'] == 'sphere' and obj_coords[0] < 0.5 or
                            obj['shape'] == 'cylinder' and obj['material'] == 'metal' and obj_coords[0] > 0.5):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                elif class_id == 4:
                    obj_coords = self.convert_coords(obj, scene['directions'])
                    if (obj['shape'] == 'cylinder' and obj['material'] == 'metal' and obj_coords[0] > 0.5):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                elif class_id == 6:
                    if (obj['shape'] == 'sphere' and obj['size'] == 'large' and obj['color'] == 'blue') or \
                            (obj['shape'] == 'sphere' and obj['size'] == 'small' and obj['color'] == 'yellow'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)

        return torch.tensor(mask) * 255 # for PIL

    def get_table_expl_mask(self, objects, class_id):
        objects = objects.T

        mask = torch.zeros(objects.shape)

        if self.conf_vers == 'conf_3':
            for i, obj in enumerate(objects):
                if class_id == 0:
                    # if cube and large
                    if (obj[3:8] == torch.tensor([0, 1, 0, 1, 0])).all():
                        mask[i, 3:8] = torch.tensor([0, 1, 0, 1, 0])
                    # or cylinder and large
                    elif (obj[3:8] == torch.tensor([0, 0, 1, 1, 0])).all():
                        mask[i, 3:8] = torch.tensor([0, 0, 1, 1, 0])
                elif class_id == 1:
                    # if cube, small, metal
                    if (obj[3:10] == torch.tensor([0, 1, 0, 0, 1, 0, 1])).all():
                        mask[i, 3:10] = torch.tensor([0, 1, 0, 0, 1, 0, 1])
                    # or sphere, small
                    elif (obj[3:8] == torch.tensor([1, 0, 0, 0, 1])).all():
                        mask[i, 3:8] = torch.tensor([1, 0, 0, 0, 1])
                elif class_id == 2:
                    # if sphere large blue
                    if ((obj[3:8] == torch.tensor([1, 0, 0, 1, 0])).all()
                        and (obj[10:] == torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])).all()).all():
                        mask[i, 3:8] = torch.tensor([1, 0, 0, 1, 0])
                        mask[i, 10:] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
                    # or sphere small yellow
                    elif ((obj[3:8] == torch.tensor([1, 0, 0, 0, 1])).all()
                          and (obj[10:] == torch.tensor([0, 0, 1, 0, 0, 0, 0, 0])).all()).all():
                        mask[i, 3:8] = torch.tensor([1, 0, 0, 0, 1])
                        mask[i, 10:] = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0])
        elif self.conf_vers == 'conf_7':
            for i, obj in enumerate(objects):
                if class_id == 0:
                    # if cube and large
                    if (obj[3:8] == torch.tensor([0, 1, 0, 1, 0])).all():
                        mask[i, 3:8] = torch.tensor([0, 1, 0, 1, 0])
                    # or cylinder and large
                    elif (obj[3:8] == torch.tensor([0, 0, 1, 1, 0])).all():
                        mask[i, 3:8] = torch.tensor([0, 0, 1, 1, 0])
                elif class_id == 1:
                    # if cube, small, metal
                    if (obj[3:10] == torch.tensor([0, 1, 0, 0, 1, 0, 1])).all():
                        mask[i, 3:10] = torch.tensor([0, 1, 0, 0, 1, 0, 1])
                    # or sphere, small
                    elif (obj[3:8] == torch.tensor([1, 0, 0, 0, 1])).all():
                        mask[i, 3:8] = torch.tensor([1, 0, 0, 0, 1])
                elif class_id == 2:
                    # get maximal y coord of red objects
                    y_red = objects[objects[:, 14] == 1, 1]
                    y_cyan = objects[objects[:, 10] == 1, 1]
                    # if cyan object and y coord greater than that of at least 2 red objs, i.e. in front of red objs
                    if ((obj[10:] == torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])).all()
                        and (sum(obj[1] > y_red) >= 2)).all():
                        mask[i, 10:] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
                        mask[i, 1] = torch.tensor([1])
                    # or red obj
                    elif ((obj[10:] == torch.tensor([0, 0, 0, 0, 1, 0, 0, 0])).all()
                          and (sum(obj[1] < y_cyan) >= 1)).all():
                        mask[i, 10:] = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0])
                        mask[i, 1] = torch.tensor([1])
                elif class_id == 3:
                    # if small and brown
                    if ((obj[6:8] == torch.tensor([0, 1])).all()
                        and (obj[10:] == torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])).all()).all():
                        mask[i, 6:8] = torch.tensor([0, 1])
                        mask[i, 10:] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])
                    # if small and green
                    elif ((obj[6:8] == torch.tensor([0, 1])).all()
                          and (obj[10:] == torch.tensor([0, 0, 0, 0, 0, 1, 0, 0])).all()).all():
                        mask[i, 6:8] = torch.tensor([0, 1])
                        mask[i, 10:] = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0])
                    # if small and purple
                    elif ((obj[6:8] == torch.tensor([0, 1])).all()
                          and (obj[10:] == torch.tensor([0, 0, 0, 1, 0, 0, 0, 0])).all()).all():
                        mask[i, 6:8] = torch.tensor([0, 1])
                        mask[i, 10:] = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0])
                    # elif small
                    elif (obj[6:8] == torch.tensor([0, 1])).all():
                        mask[i, 6:8] = torch.tensor([0, 1])
                elif class_id == 4:
                    # if at least 3 metal cylinders on right side
                    if sum((objects[:, 5] == 1) & (objects[:, 9] == 1) & (objects[:, 0] > 0.5)) >= 3:
                        # if sphere and on left side
                        if ((obj[3:6] == torch.tensor([1, 0, 0])).all()
                            and obj[0] < 0.5).all():
                            mask[i, 3:6] = torch.tensor([1, 0, 0])
                            mask[i, 0] = torch.tensor([1])
                        # if metal cyl. and on right side
                        elif ((obj[3:6] == torch.tensor([0, 0, 1])).all()
                              and (obj[8:10] == torch.tensor([0, 1])).all()
                              and obj[0] > 0.5).all():
                            mask[i, 3:6] = torch.tensor([0, 0, 1])
                            mask[i, 8:10] = torch.tensor([0, 1])
                            mask[i, 0] = torch.tensor([1])
                    # if sphere and on left side
                    elif ((obj[3:6] == torch.tensor([1, 0, 0])).all()
                          and obj[0] < 0.5).all():
                        mask[i, 3:6] = torch.tensor([1, 0, 0])
                        mask[i, 0] = torch.tensor([1])
                elif class_id == 5:
                    # if metal cylinder and on right side
                    if ((obj[3:6] == torch.tensor([0, 0, 1])).all()
                        and (obj[8:10] == torch.tensor([0, 1])).all()
                        and obj[0] > 0.5).all():
                        mask[i, 3:6] = torch.tensor([0, 0, 1])
                        mask[i, 8:10] = torch.tensor([0, 1])
                        mask[i, 0] = torch.tensor([1])
                elif class_id == 6:
                    # if sphere large blue
                    if ((obj[3:8] == torch.tensor([1, 0, 0, 1, 0])).all()
                        and (obj[10:] == torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])).all()).all():
                        mask[i, 3:8] = torch.tensor([1, 0, 0, 1, 0])
                        mask[i, 10:] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
                    # or sphere small yellow
                    elif ((obj[3:8] == torch.tensor([1, 0, 0, 0, 1])).all()
                          and (obj[10:] == torch.tensor([0, 0, 1, 0, 0, 0, 0, 0])).all()).all():
                        mask[i, 3:8] = torch.tensor([1, 0, 0, 0, 1])
                        mask[i, 10:] = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0])

        return mask

    @property
    def images_folder(self):
        return os.path.join(self.base_path, self.split, "images")

    @property
    def scenes_path(self):
        return os.path.join(
            self.base_path, self.split, "CLEVR_HANS_scenes_{}.json".format(self.split)
        )

    def __getitem__(self, item):
        image_id = self.img_ids[item]

        image = pil_loader(self.fnames[item])
        # TODO: sofar only dummy
        img_expl = torch.tensor([0])

        if self.transform is not None:
            image = self.transform(image) # in range [0., 1.]
            image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
            # img_expl = self.transform_img_expl(img_expl)

        objects = self.scenes[item]
        table_expl = self.gt_table_expls[item]
        img_class_id = self.img_class_ids[item]

        # remove objects presence indicator from gt table
        objects = objects[:, 1:]

        return image, objects, img_class_id, image_id, img_expl, table_expl

    def __len__(self):
        return len(self.scenes)


class CLEVR_HANS_EXPL_Global(torch.utils.data.Dataset):
    def __init__(self, base_path, split, conf_vers='conf_2'):
        assert split in {
            "train",
            "val",
            "test",
        }
        self.base_path = base_path
        self.split = split
        self.max_objects = 10
        self.conf_vers = conf_vers

        with open(self.scenes_path) as fd:
            scenes = json.load(fd)["scenes"]
        self.img_ids, self.img_class_ids, self.scenes, self.fnames, self.gt_img_expls, self.gt_table_expls = \
            self.prepare_scenes(scenes)

        self.transform = transforms.Compose(
            [transforms.Resize((128, 128)),
             transforms.ToTensor()]
        )

        self.n_classes = len(np.unique(self.img_class_ids))
        self.category_dict = CLASSES

        # get ids of category ranges, i.e. shape has three categories from ids 0 to 2
        self.category_ids = np.array([3, 6, 8, 10, 18])

    def convert_coords(self, obj, scene_directions):
        # Originally the x, y, z positions are in [-3, 3].
        # We re-normalize them to [0, 1].
        # convert the 3d coords based on camera position
        # conversion from ns-vqa paper, normalization for slot attention
        position = [np.dot(obj['3d_coords'], scene_directions['right']),
                    np.dot(obj['3d_coords'], scene_directions['front']),
                    obj['3d_coords'][2]]
        coords = [(p +4.)/ 8. for p in position]
        return coords

    def object_to_fv(self, obj, scene_directions):
        coords = self.convert_coords(obj, scene_directions)
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
        gt_img_expls = []
        img_class_ids = []
        gt_table_expls = []
        fnames = []
        for scene in scenes_json:
            fnames.append(os.path.join(self.images_folder, scene['image_filename']))
            img_class_ids.append(scene['class_id'])
            img_idx = scene["image_index"]

            objects = [self.object_to_fv(obj, scene['directions']) for obj in scene["objects"]]
            objects = torch.FloatTensor(objects).transpose(0, 1)

            # get gt image explanation based on the classification rule of the class label
            gt_img_expl_mask = self.get_img_expl_mask(scene)
            gt_img_expls.append(gt_img_expl_mask)

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

            # get gt table explanation based on the classification rule of the class label
            gt_table_expl_mask = self.get_table_expl_mask(objects, scene['class_id'])
            gt_table_expls.append(gt_table_expl_mask)

            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:num_objects] = 1

            # concatenate obj indication to end of object list
            objects = torch.cat((mask.unsqueeze(dim=0), objects), dim=0)

            img_ids.append(img_idx)
            scenes.append(objects.T)
        return img_ids, img_class_ids, scenes, fnames, gt_img_expls, gt_table_expls

    def get_img_expl_mask(self, scene):
        class_id = scene['class_id']

        mask = 0
        if self.conf_vers == 'conf_3':
            for obj in scene['objects']:
                if class_id == 0:
                    if (obj['shape'] == 'cube' and obj['size'] == 'large') or \
                            (obj['shape'] == 'cylinder' and obj['size'] == 'large'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                elif class_id == 1:
                    if (obj['shape'] == 'cube' and obj['size'] == 'small' and obj['material'] == 'metal') or \
                            (obj['shape'] == 'sphere' and obj['size'] == 'small'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
                elif class_id == 2:
                    if (obj['shape'] == 'sphere' and obj['size'] == 'large' and obj['color'] == 'blue') or \
                            (obj['shape'] == 'sphere' and obj['size'] == 'small' and obj['color'] == 'yellow'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)

        return torch.tensor(mask) * 255 # for PIL

    def get_table_expl_mask(self, objects, class_id):
        objects = objects.T

        mask = torch.zeros(objects.shape)

        if self.conf_vers == 'conf_3':
            for i, obj in enumerate(objects):
                if class_id == 0:
                    mask[i, 16] = torch.tensor([1])

        return mask

    @property
    def images_folder(self):
        return os.path.join(self.base_path, self.split, "images")

    @property
    def scenes_path(self):
        return os.path.join(
            self.base_path, self.split, "CLEVR_HANS_scenes_{}.json".format(self.split)
        )

    def __getitem__(self, item):
        image_id = self.img_ids[item]

        image = pil_loader(self.fnames[item])
        # only dummy
        img_expl = torch.tensor([0])

        if self.transform is not None:
            image = self.transform(image) # in range [0., 1.]
            image = (image - 0.5) * 2.0  # Rescale to [-1, 1].

        objects = self.scenes[item]
        table_expl = self.gt_table_expls[item]
        img_class_id = self.img_class_ids[item]

        # remove objects presence indicator from gt table
        objects = objects[:, 1:]

        return image, objects, img_class_id, image_id, img_expl, table_expl

    def __len__(self):
        return len(self.scenes)

