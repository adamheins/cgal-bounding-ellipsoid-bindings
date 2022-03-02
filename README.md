# CGAL Minimum Bounding Ellipsoid Bindings

This code implements Python (3.7+) bindings for the [approximate minimum volume
bounding
ellipsoid](https://doc.cgal.org/latest/Bounding_volumes/classCGAL_1_1Approximate__min__ellipsoid__d.html)
from CGAL.

## Install

First, install [CGAL](https://github.com/CGAL/cgal) 5.0+. This may be available
from your package manager, or you can do a source install using:
```
git clone https://github.com/CGAL/cgal.git
cd cgal
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCGAL_DIR:PATH=.. ..
make
sudo make install
```

Now install these bindings:
```
git clone https://github.com/adamheins/cgal-bounding-ellipsoid-bindings.git
cd cgal-bounding-ellipsoid-bindings
python3 setup.py install
```

## Usage
A 2D example is provided in `example.py` and shown below. Higher dimensions are
also supported.
```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from bounding_ellipsoid import bounding_ellipsoid


# here we use 2D points, but higher dimensions are also supported
d = 2

# generate some random points to bound
points = np.random.random((10, d)) * 10

# factor by which it is acceptable to be larger than the optimal bounding
# ellipsoid
eps = 0.01

# the bounding ellipsoid is represented by the center point, the axis radii,
# and a rotation matrix
center, radii, rotation = bounding_ellipsoid(points, eps)

# convert rotation matrix to angle
angle = np.arctan2(rotation[1, 0], rotation[0, 0])

# an ellipsoid is often expressed as the set of points x such that
#   (x - center).T @ A @ (x - center) <= 1
# we can recover the matrix A using
A = rotation @ np.diag(1.0 / radii ** 2) @ rotation.T

# test all the points for inclusion in the ellipsoid:
for i in range(points.shape[0]):
    delta = points[i, :] - center
    assert delta.T @ A @ delta <= 1, f"point {points[i, :]} is not in the ellipsoid!"


def sorted_eigs(eigvals, eigvecs):
    """Sort eigenvalues and corresponding eigenvectors from low to high.

    Eigenvectors are the columns of eigvecs.
    """
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]


# A has eigenvalues 1. / radii**2 and eigenvectors as columns of rotation;
# let's make sure this is true. Sort both to make sure they compare correctly.
eigvals, eigvecs = sorted_eigs(*np.linalg.eig(A))
inv_radii_squared, rotation = sorted_eigs(1.0 / radii ** 2, rotation)

# compare eigenvalues
assert np.allclose(eigvals, inv_radii_squared), "eigenvalues not equal!"

# compare eigenvectors: the eigenvectors are only unique up to a scale factor,
# so here we just test that they are parallel (and therefore equivalent)
for i in range(d):
    a = eigvecs[:, i]
    b = rotation[:, i]
    assert np.isclose(
        np.abs(a @ b), np.linalg.norm(a) * np.linalg.norm(b)
    ), "eigenvectors not parallel!"

# plot the points and the ellipsoid
plt.plot(points[:, 0], points[:, 1], "o")
plt.plot(center[0], center[1], "x", color="k")
ax = plt.gca()
ax.add_patch(
    mpl.patches.Ellipse(
        center,
        2 * radii[0],
        2 * radii[1],
        angle=np.rad2deg(angle),
        fill=False,
    )
)
plt.show()
```
