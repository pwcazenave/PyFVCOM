import numpy.testing as test
import numpy as np

from unittest import TestCase

from PyFVCOM.read_results import nodes2elems
from PyFVCOM.grid_tools import get_area, node_control_area, element_control_area, control_volumes


class GridToolsTest(TestCase):

    def setUp(self):
        # Make a really simple unstructured grid of 4 elements stack on top of one another.
        self.x = np.array([0, 1, 0, 1, 0, 1])
        self.y = np.array([0, 0, 1, 1, 2, 2])
        self.tri = np.array([[0, 2, 1], [1, 2, 3], [2, 5, 3], [2, 4, 5]])
        self.xc = nodes2elems(self.x, self.tri)
        self.yc = nodes2elems(self.y, self.tri)

    def tearDown(self):
        pass

    def test_get_node_control_area(self):
        test_node_area = 1 / 3
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
        test_control_volumes = [1 / 3] * len(self.x)
        test_control_volumes = [1] * len(self.x)
        node_areas, element_areas = control_volumes(self.x, self.y, self.tri)
