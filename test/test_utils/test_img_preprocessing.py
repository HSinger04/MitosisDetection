import torch
from src.utils.img_preprocessing import img_and_bboxes2patches

def test_img_and_bboxes2patches():

    img = torch.zeros((3, 9, 9))
    patch_size = 3
    bboxes = [[1, 1, 2, 2],
              [1, 1, 5, 2],
              [1, 1, 6, 2],
              [1, 1, 2, 5],
              [1, 1, 2, 6],
              [1, 1, 6, 6]]
    expected = [{0: [1, 1, 2, 2]},
                {0: [1, 1, 2, 2],
                 1: [0, 1, 3, 2]},
                {0: [1, 1, 2, 2],
                 1: [0, 1, 3, 2],
                 2: [0, 1, 1, 2]},
                {0: [1, 1, 2, 2],
                 3: [1, 0, 2, 3]},
                {0: [1, 1, 2, 2],
                 3: [1, 0, 2, 3],
                 6: [1, 0, 2, 1]},
                {0: [1, 1, 2, 2],
                 1: [0, 1, 3, 2],
                 2: [0, 1, 1, 2],
                 3: [1, 0, 2, 3],
                 4: [0, 0, 3, 3],
                 5: [0, 0, 1, 3],
                 6: [1, 0, 2, 1],
                 7: [0, 0, 3, 1],
                 8: [0, 0, 1, 1]}]

    init_bboxes = [[]] * (patch_size * patch_size)

    for bbox, exptd in zip(bboxes, expected):
        nested_bbox = [bbox]
        _, actual = img_and_bboxes2patches(img, nested_bbox, patch_size)
        for k, v in exptd.items():
            v = torch.tensor(v)
            assert torch.all(actual[k][0] == v)
            actual[k] = []
        # assert that no actual bboxes were missed
        assert actual == init_bboxes
