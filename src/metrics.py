import torch


def union(intersection:float, bx1:int, by1:int, tx1:int, ty1:int, bx2:int, by2:int, tx2:int, ty2:int) -> float:
    """Calculate union given the bottom left and top right co-ordinates of two bounding boxes

    Args:
        intersection (_type_): Intersection between the two bounding boxes.
        bx1 (int): bottom left x coordinate of first bb
        by1 (int): bottom left y coordinate of first bb
        tx1 (int): top right x coordinate of first bb
        ty1 (int): top right y coordinate of first bb
        bx2 (int): bottom left x coordinate of second bb
        by2 (int): bottom left y coordinate of second bb
        tx2 (int): top right x coordinate of second bb
        ty2 (int): top right y coordinate of second bb

    Returns:
        float: Union between the two bounding boxes
    """
    area1 = abs(bx1-tx1) * abs(by1-ty1)
    area2 = abs(bx2-tx2) * abs(by2-ty2)

    # the sum of the areas has 2*intersection already in it
    # correct by subtracting intersection

    u = (area1 + area2) - intersection

    return u


def union_t(intersection:torch.Tensor, bb1:torch.Tensor, bb2:torch.Tensor):
    """Calculate Union for pytorch Tensors

    Args:
        intersection (torch.Tensor): Intersection
        bbx1 (torch.Tensor): first bounding box (N X 4) (bx, by, tx, ty)
        bbx2 (torch.Tensor): second bounding box (N X 4) (bx, by, tx, ty)
    """

    area1 = abs(bb1[:, 0] - bb1[:, 2]) * abs(bb1[:, 1] - bb1[:, 3])
    area2 = abs(bb2[:, 0] - bb2[:, 2]) * abs(bb2[:, 1] - bb2[:, 3])

    u = (area1 + area2) - intersection

    return u


def intersection(bx1:int, by1:int, tx1:int, ty1:int, bx2:int, by2:int, tx2:int, ty2:int) -> float:
    """Calculate intersection given the bottom left and top right co-ordinates of two bounding boxes

    Args:
        bx1 (int): bottom left x coordinate of first bb
        by1 (int): bottom left y coordinate of first bb
        tx1 (int): top right x coordinate of first bb
        ty1 (int): top right y coordinate of first bb
        bx2 (int): bottom left x coordinate of second bb
        by2 (int): bottom left y coordinate of second bb
        tx2 (int): top right x coordinate of second bb
        ty2 (int): top right y coordinate of second bb

    Returns:
        float: Intersection between the two bounding boxes
    """

    ibx = max(bx1, bx2)
    iby = min(by1, by2)

    itx = min(tx1, tx2)
    ity = max(ty1, ty2)


    area = abs(ibx-itx) * abs(iby-ity)

    return (ibx, iby, itx, ity), area


def intersection_t(bb1:torch.Tensor, bb2:torch.Tensor):
    """Calculate Intersection for pytorch Tensors

    Args:
        bbx1 (torch.Tensor): first bounding box (N X 4) (bx, by, tx, ty)
        bbx2 (torch.Tensor): second bounding box (N X 4) (bx, by, tx, ty)
    """

    ibx = torch.max(bb1[:, 0], bb2[:, 0])
    iby = torch.min(bb1[:, 1], bb2[:, 1])

    itx = torch.min(bb1[:, 2], bb2[:, 2])
    ity = torch.max(bb1[:, 3], bb2[:, 3])


    area = abs(ibx-itx) * abs(iby-ity)

    return (ibx, iby, itx, ity), area


def iou(bbox1: tuple, bbox2:tuple) -> float:
    """Calculates the IoU of the two bounding boxes
    The bbox should be a tuple of (bx, by, tx, ty)

    Args:
        bbox1 (tuple): 1st bounding box
        bbox2 (tuple): 2nd bounding box

    Returns:
        float: Intersection over Union of two bb's
    """

    assert len(bbox1) == 4 and len(bbox2) == 4, 'Both bbox coordinates must be in form (bx, by, tx, ty)'

    bx1, by1, tx1, ty1 = bbox1
    bx2, by2, tx2, ty2 = bbox2

    _, i = intersection(
        bx1=bx1,
        by1=by1,
        tx1=tx1,
        ty1=ty1,

        bx2=bx2,
        by2=by2,
        tx2=tx2,
        ty2=ty2,
    )

    u = union(
        intersection=i,

        bx1=bx1,
        by1=by1,
        tx1=tx1,
        ty1=ty1,

        bx2=bx2,
        by2=by2,
        tx2=tx2,
        ty2=ty2,
    )


    IoU = i/u

    return IoU


def iou_t(bb1:torch.Tensor, bb2:torch.Tensor) -> torch.Tensor:
    _, i = intersection_t(bb1, bb2)
    u = union_t(i, bb1, bb2)

    IoU = i/u

    return IoU


def MeanAveragePrecision(bb1:torch.Tensor, bb2:torch.Tensor) -> float:
    pass



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    bbox1 = (0, 6, 5, 0)
    bbox2 = (3, 9, 9, 3)

    (ibx, iby, itx, ity), _ = intersection(*bbox1, *bbox2)


    print(f'Intersection @ bottom=>({ibx}, {iby}) top=>({itx}, {ity})')

    IoU = iou(bbox1, bbox2)

    bbox1_t = torch.Tensor([list(bbox1), list(bbox1)])
    bbox2_t = torch.Tensor([list(bbox2), list(bbox2)])

    IoU_t = iou_t(bbox1_t, bbox2_t)

    assert IoU_t[0].item() == IoU_t[1].item()
    assert IoU_t[0].item() - IoU < 1e-5

    print(IoU, IoU_t)

    img = np.zeros((10, 10, 3))

    for i in range(10):
        for j in range(10):

            ## draw first bb

            if i == 0:
                if j <6:
                    img[0, j, :] = [255, 0, 0]#[0][j] = 1

            if j == 0 or j == 5:
                if i < 6:
                    img[i, j, :] = [255, 0, 0]

            if i == 6:
                if j<6:
                    img[i, j, :] = [255, 0, 0]


            ## draw second bb

            if i ==3:
                if j >= 3 and j<10:
                    img[i, j, :] = [0, 255, 0]

            if j == 3 or j == 9:
                if i >2:
                    img[i, j, :] = [0, 255, 0]

            if i == 9:
                if j>3:
                    img[i, j, :] = [0, 255, 0]

    plt.imshow(img)
    plt.show()


    