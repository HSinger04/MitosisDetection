def test_img_and_bboxes2patches():

    img = torch.zeros((3, 9, 9))
    patch_size = 3
    bboxes = [[2, 2, 1, 1],
              [2, 2, 3, 1],
              [2, 2, 5, 1],
              [2, 2, 1, 3],
              [2, 2, 1, 5],
              [2, 2, 5, 5]]
    expected = [{0: [2, 2, 1, 1]},
                {0: [2, 2, 1, 1],
                 1: [0, 2, 2, 1]},
                {0: [2, 2, 1, 1],
                 1: [0, 2, 3, 1],
                 2: [0, 2, 1, 1]},
                {0: [2, 2, 1, 1],
                 3: [2, 0, 1, 2]},
                {0: [2, 2, 1, 1],
                 3: [2, 0, 1, 3],
                 6: [2, 0, 1, 1]},
                {0: [2, 2, 1, 1],
                 1: [0, 2, 3, 1],
                 2: [0, 2, 1, 1],
                 3: [2, 0, 1, 3],
                 4: [0, 0, 3, 3],
                 5: [0, 0, 1, 3],
                 6: [2, 0, 1, 1],
                 7: [0, 0, 3, 1],
                 8: [0, 0, 1, 1]}]

    init_bboxes = [[]] * (patch_size * patch_size)

    for bbox, exptd in zip(bboxes, expected):
        nested_bbox = [bbox]
        _, actual = img_and_bboxes2patches(img, nested_bbox, patch_size)
        for k, v in exptd.items():
            print("\n" * 2)
            print(actual[k])
            print("\n")
            print(v)
            assert actual[k] == v
            actual[k] = []
        # assert that no actual bboxes were missed
        assert actual == init_bboxes