import math
import unittest

from qsol_geo_reason.geometry import align_pair, cosine_alignment, finite_difference, menger_curvature_sequence, path_length, resample_arclength


class GeometryTests(unittest.TestCase):
    def test_finite_differences(self):
        points = [[0.0], [1.0], [3.0], [6.0]]
        self.assertEqual(finite_difference(points, 0), points)
        self.assertEqual(finite_difference(points, 1), [[1.0], [2.0], [3.0]])
        self.assertEqual(finite_difference(points, 2), [[1.0], [1.0]])
        self.assertEqual(finite_difference(points, 4), [])

    def test_path_length(self): self.assertAlmostEqual(path_length([[0, 0], [3, 4], [6, 8]]), 10.0)

    def test_null_path_and_zero_vectors(self):
        null = [[2, 2], [2, 2], [2, 2]]
        self.assertEqual(path_length(null), 0.0)
        self.assertEqual(menger_curvature_sequence(null), [0.0])
        self.assertEqual(cosine_alignment(null, null, order=1, align="error"), 1.0)

    def test_menger_curvature_circle(self):
        r = 2.0
        pts = [[r * math.cos(a), r * math.sin(a)] for a in [0, math.pi/4, math.pi/2]]
        self.assertAlmostEqual(menger_curvature_sequence(pts)[0], 1.0 / r, places=12)

    def test_collinear_curvature_zero(self): self.assertEqual(menger_curvature_sequence([[0, 0], [1, 0], [2, 0]]), [0.0])
    def test_repeated_point_curvature_zero(self): self.assertEqual(menger_curvature_sequence([[0, 0], [0, 0], [1, 0]]), [0.0])
    def test_short_sequence(self): self.assertEqual(menger_curvature_sequence([[0, 0], [1, 0]]), [])

    def test_dimension_mismatch_fails(self):
        with self.assertRaises(ValueError): finite_difference([[0, 0], [1]])
        with self.assertRaises(ValueError): align_pair([[0, 0]], [[0, 0, 0]])

    def test_arclength_resampling(self):
        self.assertEqual(resample_arclength([[0, 0], [2, 0]], 5), [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]])

    def test_arclength_resampling_degenerate(self): self.assertEqual(resample_arclength([[1, 2], [1, 2]], 3), [[1.0, 2.0]] * 3)

    def test_alignment_modes(self):
        a, b = [[0, 0], [1, 0], [2, 0]], [[0, 0], [2, 0]]
        x, y = align_pair(a, b, mode="truncate")
        self.assertEqual(len(x), len(y))
        with self.assertRaises(ValueError): align_pair(a, b, mode="error")
        x, y = align_pair(a, b, mode="arclength")
        self.assertEqual(x, y)
