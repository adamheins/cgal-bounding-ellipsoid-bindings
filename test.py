from bounding_ellipsoid import bounding_ellipsoid
import numpy as np
import IPython


P = np.random.random((10, 2))
c, L, R = bounding_ellipsoid(P, 0.01)
IPython.embed()
