import numpy.testing as test
import numpy as np

from unittest import TestCase

from PyFVCOM.read_results import nodes2elems
from PyFVCOM.grid_tools import get_area, node_control_area, element_control_area, control_volumes


class GridToolsTest(TestCase):

    def setUp(self):
        """ Make a really simple unstructured grid of 8 elements as two rows of 4 elements. """
        self.x = np.array([0, 1, 0, 1, 0, 1, 2, 2, 2])
        self.y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        self.tri = np.array([[0, 2, 1], [1, 2, 3], [2, 5, 3], [2, 4, 5], [1, 3, 7], [1, 7, 6], [3, 5, 7], [7, 5, 8]])
        self.xc = nodes2elems(self.x, self.tri)
        self.yc = nodes2elems(self.y, self.tri)
        self.z = np.array([0, 1, 1, 0, 2, 1, 2, 3, 3])

    def test_get_node_control_area(self):
        test_node_area = 2 / 3
        node = 1
        node_area = node_control_area(node, self.x, self.y, self.xc, self.yc, self.tri)
        test.assert_almost_equal(test_node_area, node_area)

    def test_get_element_control_area(self):
        test_element_area = 2
        node = 2
        art = get_area(np.asarray((self.x[self.tri[:, 0]], self.y[self.tri[:, 0]])).T,
                       np.asarray((self.x[self.tri[:, 1]], self.y[self.tri[:, 1]])).T,
                       np.asarray((self.x[self.tri[:, 2]], self.y[self.tri[:, 2]])).T)
        element_area = element_control_area(node, self.tri, art)
        test.assert_almost_equal(test_element_area, element_area)

    def test_get_control_volumes(self):
        test_node_areas = [1 / 6, 2 / 3, 2 / 3,
                           2 / 3, 1 / 6, 2 / 3,
                           1 / 6, 2 / 3, 1 / 6]
        test_element_areas = [0.5, 2, 2, 2, 0.5, 2, 0.5, 2, 0.5]
        node_areas, element_areas = control_volumes(self.x, self.y, self.tri)