def test_img_and_bboxes2patches():

    img = torch.zeros((3, 9, 9))
    patch_size = 3
    bboxes = [[1, 1, 1, 1],
              [1, 1, 4, 1],
              [1, 1, 5, 1],
              [1, 1, 1, 4],
              [1, 1, 1, 5],
              [1, 1, 5, 5]]
    expected = [{0: [1, 1, 1, 1]},
                {0: [1, 1, 1, 1],
                 1: [0, 1, 2, 1]},
                {0: [1, 1, 1, 1],
                 1: [0, 1, 2, 1],
                 2: [0, 1, 0, 1]},
                {0: [1, 1, 1, 1],
                 3: [1, 0, 1, 2]},
                {0: [1, 1, 1, 1],
                 3: [1, 0, 1, 2],
                 6: [1, 0, 1, 0]},
                {0: [1, 1, 1, 1],
                 1: [0, 1, 2, 1],
                 2: [0, 1, 0, 1],
                 3: [1, 0, 1, 2],
                 4: [0, 0, 2, 2],
                 5: [0, 0, 0, 2],
                 6: [1, 0, 1, 0],
                 7: [0, 0, 2, 0],
                 8: [0, 0, 0, 0]}]

    init_bboxes = [[]] * (patch_size * patch_size)

    for bbox, exptd in zip(bboxes, expected):
        nested_bbox = [bbox]
        _, actual = img_and_bboxes2patches(img, nested_bbox, patch_size)
        for k, v in exptd.items():
            assert actual[k][0] == v
            actual[k] = []
        # assert that no actual bboxes were missed
        assert actual == init_bboxes
