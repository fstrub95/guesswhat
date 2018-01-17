import gzip
import json
import copy
import numpy as np

from generic.data_provider.dataset import AbstractDataset

use_100 = True

try:
    import cocoapi.PythonAPI.pycocotools.mask as cocoapi

    use_coco = True
except ImportError:
    print("Coco API was not detected - advanced segmentation features cannot be used")
    use_coco = False
    pass


class Game:

    def __init__(self, id, object_id, image, objects, qas, status, which_set, image_builder, crop_builder):
        self.dialogue_id = id
        self.object_id = object_id
        self.image = Image(id=image["id"],
                           width=image["width"],
                           height=image["height"],
                           url=image["coco_url"],
                           which_set=which_set,
                           image_builder=image_builder)
        self.objects = []
        for o in objects:

            new_obj = Object(id=o['id'],
                             category=o['category'],
                             category_id=o['category_id'],
                             bbox=Bbox(o['bbox'], image["width"], image["height"]),
                             area=o['area'],
                             segment=o['segment'],
                             crop_builder=crop_builder,
                             which_set=which_set,
                             image=self.image)

            self.objects.append(new_obj)
            if o['id'] == object_id:
                self.object = new_obj  # Keep ref on the object to find

        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status


class Image:
    def __init__(self, id, width, height, url, which_set, image_builder=None):
        self.id = id
        self.width = width
        self.height = height
        self.url = url

        self.image_loader = None
        if image_builder is not None:
            filename = "{}.jpg".format(id)
            self.image_loader = image_builder.build(id, which_set=which_set, filename=filename, optional=False)

    def get_image(self, **kwargs):
        if self.image_loader is not None:
            return self.image_loader.get_image(**kwargs)
        else:
            return None


class Bbox:
    def __init__(self, bbox, im_width, im_height):
        # Retrieve features (COCO format)
        self.x_width = bbox[2]
        self.y_height = bbox[3]

        self.x_left = bbox[0]
        self.x_right = self.x_left + self.x_width

        self.y_upper = im_height - bbox[1]
        self.y_lower = self.y_upper - self.y_height

        self.x_center = self.x_left + 0.5 * self.x_width
        self.y_center = self.y_lower + 0.5 * self.y_height

        self.coco_bbox = bbox


class Object:
    def __init__(self, id, category, category_id, bbox, area, segment, crop_builder, image, which_set):
        self.id = id
        self.category = category
        self.category_id = category_id
        self.bbox = bbox
        self.area = area
        self.segment = segment

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        self.rle_mask = None
        if use_coco:
            self.rle_mask = cocoapi.frPyObjects(self.segment,
                                                h=image.height,
                                                w=image.width)

        if crop_builder is not None:
            filename = "{}.jpg".format(image.id)
            self.crop_loader = crop_builder.build(id, filename=filename, which_set=which_set, bbox=bbox)
            self.crop_scale = crop_builder.scale

    def get_mask(self):
        assert self.rle_mask is not None, "Mask option are not available, please compile and link cocoapi (cf. cocoapi/PythonAPI/setup.py)"
        tmp_mask = cocoapi.decode(self.rle_mask)
        if len(tmp_mask.shape) > 2:  # concatenate several mask into a single one
            tmp_mask = np.sum(tmp_mask, axis=2)
            tmp_mask[tmp_mask > 1] = 1

        return tmp_mask.astype(np.float32)

    def get_crop(self, **kwargs):
        assert self.crop_loader is not None, "Invalid crop loader"
        return self.crop_loader.get_image(**kwargs)


class Dataset(AbstractDataset):
    """Loads the dataset."""

    def __init__(self, folder, which_set, image_builder=None, crop_builder=None):
        file = '{}/guesswhat.{}.jsonl.gz'.format(folder, which_set)
        games = []

        self.set = which_set

        with gzip.open(file) as f:
            for line in f:
                line = line.decode("utf-8")
                game = json.loads(line.strip('\n'))

                g = Game(id=game['id'],
                         object_id=game['object_id'],
                         objects=game['objects'],
                         qas=game['qas'],
                         image=game['image'],
                         status=game['status'],
                         which_set=which_set,
                         image_builder=image_builder,
                         crop_builder=crop_builder)

                games.append(g)

                if use_100 and len(games) > 200: break

        super(Dataset, self).__init__(games)


# class OracleDataset(AbstractDataset):
#     """
#     Each game contains a single question answer pair
#     """
#     def __init__(self, dataset, split_question=True):
#         old_games = dataset.get_data()
#         new_games = []
#
#         for g in old_games:
#             if split_question:
#                 new_games += self.independant_split(g)
#             else:
#                 new_games += self.contiguous_split(g)
#
#         super(OracleDataset, self).__init__(new_games)
#
#     @classmethod
#     def load(cls, folder, which_set, image_builder=None, crop_builder=None, split_question=True):
#         return cls(Dataset(folder, which_set, image_builder, crop_builder), split_question)
#
#     def independant_split(self, game):
#         games = []
#         for i, q, a in zip(game.question_ids, game.questions, game.answers):
#             new_game = copy.copy(game)
#             new_game.questions = [q]
#             new_game.question_ids = [i]
#             new_game.answers = [a]
#             games.append(new_game)
#         return games
#
#     def contiguous_split(self, game):
#         games = []
#         for i in range(len(game.question_ids)):
#             new_game = copy.copy(game)
#             new_game.questions = game.questions[:i + 1]
#             new_game.question_ids = game.question_ids[:i + 1]
#             new_game.answers = game.answers[:i + 1]
#             games.append(new_game)
#         return games


class CropDataset(AbstractDataset):
    """
    Each game contains no question/answers but a new object
    """

    def __init__(self, dataset, expand_objects):
        old_games = dataset.get_data()
        new_games = []

        for g in old_games:
            if expand_objects:
                new_games += self.split(g)
            else:
                new_games += self.update_ref(g)
        super(CropDataset, self).__init__(new_games)

    @classmethod
    def load(cls, folder, which_set, image_builder=None, crop_builder=None, expand_objects=False):
        return CropDataset(Dataset(folder, which_set, image_builder, crop_builder), expand_objects=expand_objects)

    def split(self, game):
        games = []
        for obj in game.objects:
            new_game = copy.copy(game)
            new_game.questions = [""]
            new_game.question_ids = [0]
            new_game.answers = [""]
            new_game.object_id = obj.id

            # update object reference
            new_game.object = [o for o in game.objects if o.id == obj.id][0]

            # Hack the image id to differentiate objects
            new_game.image = copy.copy(game.image)
            new_game.image.id = obj.id

            games.append(new_game)

        return games

    def update_ref(self, game):

        new_game = copy.copy(game)
        new_game.questions = [""]
        new_game.question_ids = [0]
        new_game.answers = [""]

        # Hack the image id to differentiate objects
        new_game.image = copy.copy(game.image)
        new_game.image.id = game.object_id

        return [new_game]


def dump_samples_into_dataset(data, save_path, tokenizer, name="model", true_id=False):
    with gzip.open(save_path.format('guesswhat.' + name + '.jsonl.gz'), 'wb') as f:
        for _, d in enumerate(data):
            dialogue = d["dialogue"]
            game = d["game"]
            object_id = d["object_id"]
            success = d["success"]
            prob_objects = d["prob_objects"]
            guess_object_id = d["guess_object_id"]

            sample = {}

            qas = []
            start = 1
            for k, word in enumerate(dialogue):
                if word == tokenizer.yes_token or \
                        word == tokenizer.no_token or \
                        word == tokenizer.non_applicable_token:
                    q = tokenizer.decode(dialogue[start:k - 1])
                    a = tokenizer.decode([dialogue[k]])

                    prob_obj = list(prob_objects[len(qas), :len(game.objects)])
                    prob_obj = [str(round(p, 3)) for p in prob_obj]  # decimal are not supported y default in json encoder

                    qas.append({"question": q,
                                "answer": a[1:-1],
                                "id": k,
                                "p": prob_obj})

                    start = k + 1

            sample["id"] = game.dialogue_id if true_id else 0
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment": [],  # no segment to avoid making the file to big
                                  } for o in game.objects]

            sample["object_id"] = object_id
            sample["guess_object_id"] = guess_object_id
            sample["status"] = "success" if success else "failure"

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')


def dump_oracle(oracle_data, games, save_path, name="oracle"):
    with gzip.open(save_path.format('guesswhat.' + name + '.jsonl.gz'), 'wb') as f:
        for game in games:

            qas = oracle_data[game.dialogue_id]
            sample = {}

            # check that question/answer are correctly sorted
            for qa, q_id in zip(qas, game.question_ids):
                assert qa["id"] == q_id

            for qo, qh in zip(qas, game.questions):
                assert qo["question"] == qh, "{} vs {}".format(qo, qh)

            sample["id"] = game.dialogue_id
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment": o.segment,
                                  } for o in game.objects]

            sample["object_id"] = game.object_id
            sample["guess_object_id"] = game.object_id
            sample["status"] = game.status

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')
