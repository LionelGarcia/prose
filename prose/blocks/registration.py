import astroalign
import numpy as np
from scipy.spatial import KDTree
from prose.blocks.base import Block


def distance(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))


def distances(coords, coord):
    return [
        np.sqrt(((coord[0] - x)**2 + (coord[1] - y)**2))
        for x, y in zip(coords[0].flatten(), coords[1].flatten())
    ]


def clean_stars_positions(positions, tolerance=50, output_id=False):
    keep = []

    distance_to_others = np.array(
        [[distance(v, w) for w in positions] for v in positions]
    )
    for i, _distances in enumerate(distance_to_others):
        _distances[i] = np.inf
        close_stars = np.argwhere(_distances < tolerance)
        if len(close_stars) > 0:
            keep.append(np.min([close_stars[0][0], i]))
        else:
            keep.append(i)

    if output_id:
        return positions[np.unique(keep)], np.unique(keep)
    else:
        return positions[np.unique(keep)]


def xyshift(im_stars_pos, ref_stars_pos, tolerance=1.5, clean=False):
    """
    Compute shift between two set of coordinates (e.g. stars)

    Parameters
    ----------
    im_stars_pos : list or ndarray
        (x,y) coordinates of n points (shape should be (2, n))
    ref_stars_pos : list or ndarray
        [(x,y) coordinates of n points (shape should be (2, n)). Reference set
    tolerance : float, optional
        by default 1.5
    clean : bool, optional
        Merge coordinates if too close, by default False

    Returns
    -------
    ndarray
        (dx, dy) shift
    """
    assert len(im_stars_pos) > 2, f"{len(im_stars_pos)} star coordinates provided (should be > 2)"

    if clean:
        clean_ref = clean_stars_positions(ref_stars_pos)
        clean_im = clean_stars_positions(im_stars_pos)
    else:
        clean_ref = ref_stars_pos
        clean_im = im_stars_pos

    delta_x = np.array([clean_ref[:, 0] - v for v in clean_im[:, 0]]).flatten()
    delta_y = np.array([clean_ref[:, 1] - v for v in clean_im[:, 1]]).flatten()

    delta_x_compare = []
    for i, dxi in enumerate(delta_x):
        dcxi = dxi - delta_x
        dcxi[i] = np.inf
        delta_x_compare.append(dcxi)

    delta_y_compare = []
    for i, dyi in enumerate(delta_y):
        dcyi = dyi - delta_y
        dcyi[i] = np.inf
        delta_y_compare.append(dcyi)

    tests = [
        np.logical_and(np.abs(dxc) < tolerance, np.abs(dyc) < tolerance)
        for dxc, dyc in zip(delta_x_compare, delta_y_compare)
    ]
    num = np.array([np.count_nonzero(test) for test in tests])

    max_count_num_i = int(np.argmax(num))
    max_nums_ids = np.argwhere(num == num[max_count_num_i]).flatten()
    dxs = np.array([delta_x[np.where(tests[i])] for i in max_nums_ids])
    dys = np.array([delta_y[np.where(tests[i])] for i in max_nums_ids])

    return np.nan_to_num(np.array([np.mean(dxs), np.mean(dys)]))


def astroalign_optimized_find_transform(
    source, target_controlp, target_invariant_tree, target_asterisms
):
    """
    A faster version of astroalign.find_transform considering that we know the
    target control points, invariants tree and asterisms. For details, see
    astroalign.find_transform

    This allows to compute control points once for reference frame
    """

    source_controlp = astroalign._find_sources(source)[: astroalign.MAX_CONTROL_POINTS]

    # Check for low number of reference points
    if len(source_controlp) < 3:
        raise Exception(
            "Reference stars in source image are less than the " "minimum value (3)."
        )
    if len(target_controlp) < 3:
        raise Exception(
            "Reference stars in target image are less than the " "minimum value (3)."
        )

    source_invariants, source_asterisms = astroalign._generate_invariants(
        source_controlp
    )
    source_invariant_tree = KDTree(source_invariants)
    matches_list = source_invariant_tree.query_ball_tree(target_invariant_tree, r=0.1)

    matches = []
    for t1, t2_list in zip(source_asterisms, matches_list):
        for t2 in target_asterisms[t2_list]:
            matches.append(list(zip(t1, t2)))
    matches = np.array(matches)

    inv_model = astroalign._MatchTransform(source_controlp, target_controlp)
    n_invariants = len(matches)
    max_iter = n_invariants
    min_matches = max(1, min(10, int(n_invariants * astroalign.MIN_MATCHES_FRACTION)))
    if (len(source_controlp) == 3 or len(target_controlp) == 3) and len(matches) == 1:
        best_t = inv_model.fit(matches)
        inlier_ind = np.arange(len(matches))  # All of the indices
    else:
        best_t, inlier_ind = astroalign._ransac(
            matches, inv_model, 1, max_iter, astroalign.PIXEL_TOL, min_matches
        )
    triangle_inliers = matches[inlier_ind]
    d1, d2, d3 = triangle_inliers.shape
    inl_arr = triangle_inliers.reshape(d1 * d2, d3)
    inl_unique = set(tuple(pair) for pair in inl_arr)
    inl_arr_unique = np.array(list(list(apair) for apair in inl_unique))
    so, d = inl_arr_unique.T

    return best_t, (source_controlp[so], target_controlp[d])


class Registration(Block):

    def __init__(self, detection=None, **kwargs):
        super().__init__(**kwargs)
        self.reference_stars = None
        self.detection = detection


class DistanceRegistration(Registration):
    def __init__(self, tolerance=1.5, clean=False, detection=None, reference=1/2, **kwargs):
        super().__init__(detection=detection, **kwargs)
        self.tolerance = tolerance
        self.clean = clean
        self.reference = reference


class XYShift(Registration):
    """
    Compute the linear shift between two point clouds. Star coordinates in the image are expected in image.stars_coords

    Parameters
    ----------
    reference : [type]
        (x,y) coordinates of n reference stars (shape should be (2, n))
    tolerance : float, optional
        maximum difference for which shifts are considered similar, by default 1.5
    clean : bool, optional
        Merge coordinates if too close, by default False
    """

    def __init__(self, reference, tolerance=1.5, clean=False, **kwargs):

        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.clean = clean
        self.reference = reference

    def run(self, image, **kwargs):
        shift = xyshift(image.stars_coords, self.reference, tolerance=self.tolerance, clean=self.clean)
        image.shift = shift
        image.header["DX"] = shift[0]
        image.header["DY"] = shift[1],
        image.header["ALIGNALG"] = self.__class__.__name__

    @staticmethod
    def doc():
        return r"""**xyshift** is originaly the **TRAPHOT** (M. Gillon) method to align consecutive images.

*Principle*: Let's consider two images, ``im`` and ``ref``. We want to know the shift between ``im`` and ``ref`` knowing the position of the stars in these two images. We suppose that there is N stars in ``im`` and ``ref`` and that their position are given by

.. math::

   \boldsymbol{S^{im}} = \begin{bmatrix}
   x^{im}_0 & y^{im}_0 \\
   x^{im}_1 & y^{im}_1 \\
   . & . \\
   x^{im}_N & y^{im}_N
   \end{bmatrix} \quad and \quad 
   \boldsymbol{S^{ref}} = \begin{bmatrix}
   x^{ref}_0 & y^{ref}_0 \\
   x^{ref}_1 & y^{ref}_1 \\
   . & . \\
   x^{ref}_N & y^{ref}_N
   \end{bmatrix}


:math:`x_i, y_i` being the coordinates of the star :math:`i`

If we want to know, let's say, the :math:`x` shift between `im` and `ref` we just have to compute :math:`x^{im}_0 - x^{ref}_0`, or to be more accurate, by considering all the stars, the mean shift

.. math::

    \Delta x = \frac{1}{N}\sum_{0}^{N} x^{im}_i - x^{ref}_i


This is possible because :math:`x^{im}_i`  and :math:`x^{ref}_i` are the position of the same identified star :math:`i` (:math:`N` of them in each image)

Identifying stars in the sky, for example using their position with respect to constellations or by trying to match catalogs (as with astrometry.net) is possible with a single image, but is time-consuming when applied to a complete set of images. Unfortunately most of the time detected stars do not hold the same index :math:`i` (index is attributed by the star detection algorithm) and their number might differ.

The goal of **xyshift** is to check all the possible shifts from one star with respect to all the others, and to identify the most common shifts between stars of two different images.

.. image:: images/xyshift_principle.png"""


class AstroAlignShift(Registration):
    """
    Compute the linear shift between point clouds using :code:`astroalign`
    """

    def __init__(self, reference_stars, **kwargs):
        super().__init__(**kwargs)
        self.reference_invariants = None
        self.reference_asterisms = None
        self.set_reference(reference_stars)

    def set_reference(self, reference_stars):
        print("AstroAlignShift overwrites detection method")
        self.reference_stars = reference_stars[:, astroalign.MAX_CONTROL_POINTS]
        self.reference_invariants, self.reference_asterisms = astroalign._generate_invariants(
            self.reference_stars)

        # self.reference.stars_coords = self.reference.stars_coords[:astroalign.MAX_CONTROL_POINTS]
        # self.reference_invariants, self.reference_asterisms = astroalign._generate_invariants(
        #     self.reference.stars_coords)

    def run(self, image, **kwargs):
        transform, _detected_stars = astroalign_optimized_find_transform(
                    image.stars_coords,
                    self.reference_stars,
                    KDTree(self.reference_invariants),
                    self.reference_asterisms,
                )
        shift = transform.translation
        image.shift = shift
        image.header["DX"] = shift[0]
        image.header["DY"] = shift[1],
        image.header["ALIGNALG"] = self.__class__.__name__


    def citations(self):
        return "astroalign"

    @staticmethod
    def doc():
        return """`astroalign <https://astroalign.readthedocs.io/en/latest/>`_ is a python module used to align stellar astronomical images using 3-point asterisms (triangles) similarities. For speed, reference asterisms are computed once at the begining of the reduction and then matched with every images."""
