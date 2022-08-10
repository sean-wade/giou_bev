import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d
# import numba


def box2corners3d(bbox):
    """ the coordinates for bottom corners
    """
    x,y,z,l,w,h,o = bbox
    center = np.array([x, y, z])
    bottom_corners = np.array(box2corners2d(bbox))
    up_corners = 2 * center - bottom_corners
    corners = np.concatenate([up_corners, bottom_corners], axis=0)
    return corners.tolist()


def box2corners2d(bbox):
    """ the coordinates for bottom corners
    """
    x,y,z,l,w,h,o = bbox
    bottom_center = np.array([x, y, z - h / 2])
    cos, sin = np.cos(o), np.sin(o)
    pc0 = np.array([x + cos * l / 2 + sin * w / 2,
                    y + sin * l / 2 - cos * w / 2,
                    z - h / 2])
    pc1 = np.array([x + cos * l / 2 - sin * w / 2,
                    y + sin * l / 2 + cos * w / 2,
                    z - h / 2])
    pc2 = 2 * bottom_center - pc0
    pc3 = 2 * bottom_center - pc1

    return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]


def iou2d(box_a, box_b):
    boxa_corners = np.array(box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap = reca.intersection(recb).area
    area_a = reca.area
    area_b = recb.area
    iou = overlap / (area_a + area_b - overlap + 1e-10)
    return iou


def iou3d(box_a, box_b):
    boxa_corners = np.array(box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap_area = reca.intersection(recb).area
    iou_2d = overlap_area / (reca.area + recb.area - overlap_area)

    ha, hb = box_a[5], box_b[5]
    za, zb = box_a[2], box_b[2]
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    overlap_volume = overlap_area * overlap_height
    union_volume = box_a[4] * box_a[3] * ha + box_b[4] * box_b[3] * hb - overlap_volume
    iou_3d = overlap_volume / (union_volume + 1e-5)

    return iou_2d, iou_3d


def giou2d(box_a, box_b):
    boxa_corners = np.array(box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    # compute intersection and union
    I = reca.intersection(recb).area
    U = box_a[4] * box_a[3] + box_b[4] * box_b[3] - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    # print("all_corners : ", all_corners)
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area

    # print("I, U, C, convex_area = ", I, U, C, convex_area)
    # compute giou
    return I / U - (C - U) / C


def giou3d(box_a, box_b):
    boxa_corners = np.array(box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    ha, hb = box_a[5], box_b[5]
    za, zb = box_a[2], box_b[2]
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    union_height = max((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2))

    # compute intersection and union
    I = reca.intersection(recb).area * overlap_height
    U = box_a[4] * box_a[3] * ha + box_b[4] * box_b[3] * hb - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area * union_height

    # print("I, U, C, convex_area, union_height = ", I, U, C, convex_area, union_height)
    # compute giou
    giou = I / U - (C - U) / C
    return giou


def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area


# @numba.njit
def iouMatrix(boxesA, boxesB):
    iou_matrix = np.ones((len(boxesA), len(boxesB))) * -100.0
    for i, d in enumerate(boxesA):
        for j, g in enumerate(boxesB):
            iou_matrix[i, j] = giou3d(d, g)
    return iou_matrix


if __name__ == "__main__":
    boxa = [10, 20, 30, 4, 3, 1, 0]
    boxb = [10, 20, 31, 4.2, 3, 1, np.pi/2.0]

    print(box2corners3d(boxa))
    print(box2corners3d(boxb))
    print(iou2d(boxa, boxb))
    print(iou3d(boxa, boxb))
    print(giou2d(boxa, boxb))
    print(giou3d(boxa, boxb))

    # boxesA = np.array([boxa,
    #                     boxa,
    #                     boxa,
    #                     boxa,
    #                     boxa,
    #                     boxa,
    #                     boxa,
    #                     boxa,
    #                     ])
    # boxesB = np.array([boxb,
    #                     boxa,
    #                     boxb,
    #                     boxb,
    #                     boxb,
    #                     boxb,
    #                     boxb,
    #                     boxa,
    #                     ])
    # print(iouMatrix(boxesA, boxesB))
