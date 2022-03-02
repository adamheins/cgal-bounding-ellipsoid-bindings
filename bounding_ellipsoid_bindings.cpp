#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <CGAL/Approximate_min_ellipsoid_d.h>
#include <CGAL/Approximate_min_ellipsoid_d_traits_d.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/MP_Float.h>
#include <CGAL/point_generators_d.h>

#include <vector>

typedef CGAL::Cartesian_d<double> Kernel;
typedef CGAL::MP_Float ET;
typedef CGAL::Approximate_min_ellipsoid_d_traits_d<Kernel, ET> Traits;
typedef Traits::Point Point;
typedef CGAL::Approximate_min_ellipsoid_d<Traits> AME;

struct EllipsoidData {
    EllipsoidData(){};

    EllipsoidData(const EllipsoidData &other) {
        center = other.center;
        radii = other.radii;
        rotation = other.rotation;
    }

    ~EllipsoidData() = default;

    std::vector<double> center;
    std::vector<double> radii;
    std::vector<double> rotation;
};

EllipsoidData bounding_ellipsoid(const double *data, const size_t n,
                                 const size_t d, double eps) {
    // Construct the points
    std::vector<Point> points;
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> point(d);
        for (size_t j = 0; j < d; ++j) {
            point[j] = data[j + d * i];
        }
        points.push_back(Point(d, point.begin(), point.end()));
    }

    // Compute the bounding ellipsoid
    Traits traits;
    AME ame(eps, points.begin(), points.end(), traits);

    // Copy center point
    EllipsoidData ellipsoid;
    for (AME::Center_coordinate_iterator c_it = ame.center_cartesian_begin();
         c_it != ame.center_cartesian_end(); ++c_it) {
        ellipsoid.center.push_back(*c_it);
    }

    // Copy axis radii and directions
    AME::Axes_lengths_iterator axes = ame.axes_lengths_begin();
    for (size_t i = 0; i < d; ++i) {
        ellipsoid.radii.push_back(*axes++);
        for (AME::Axes_direction_coordinate_iterator d_it =
                 ame.axis_direction_cartesian_begin(i);
             d_it != ame.axis_direction_cartesian_end(i); ++d_it) {
            ellipsoid.rotation.push_back(*d_it);
        }
    }

    // Transpose the rotation matrix, such that the ellipsoid matrix A can be
    // computed using A = R * D * R.T, where D = diag(radii^2)^{-1}.
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < i; ++j) {
            double temp = ellipsoid.rotation[i * d + j];
            ellipsoid.rotation[i * d + j] = ellipsoid.rotation[j * d + i];
            ellipsoid.rotation[j * d + i] = temp;
        }
    }

    return ellipsoid;
}

static PyObject *bounding_ellipsoid_python(PyObject *self, PyObject *args) {
    // We expect a 2-d array of points and a double eps representing bound
    // tolerance
    PyArrayObject *arr;
    double eps;
    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &arr, &eps)) {
        return NULL;
    }

    int ndim = PyArray_NDIM(arr);
    if (ndim != 2) {
        return NULL;
    }

    // Extract required data
    double *data = (double *)PyArray_DATA(arr);
    npy_intp *dims = PyArray_DIMS(arr);
    int n = (int)dims[0];
    int d = (int)dims[1];

    // Compute the ellipsoid
    EllipsoidData *ellipsoid =
        new EllipsoidData(bounding_ellipsoid(data, n, d, eps));

    // Convert everything to Python objects and return as the tuple (center,
    // radius).
    npy_intp vec_dims[] = {d};
    npy_intp mat_dims[] = {d, d};
    PyObject *center_array = PyArray_SimpleNewFromData(
        1, vec_dims, NPY_DOUBLE, ellipsoid->center.data());
    PyObject *radii_array = PyArray_SimpleNewFromData(
        1, vec_dims, NPY_DOUBLE, ellipsoid->radii.data());
    PyObject *rotation_array = PyArray_SimpleNewFromData(
        2, mat_dims, NPY_DOUBLE, ellipsoid->rotation.data());

    // Return a tuple of ellipsoid data
    PyObject *res = PyTuple_New(3);
    PyTuple_SetItem(res, 0, center_array);
    PyTuple_SetItem(res, 1, radii_array);
    PyTuple_SetItem(res, 2, rotation_array);
    return res;
}

static PyMethodDef BoundingEllipsoidMethods[] = {
    {"bounding_ellipsoid", bounding_ellipsoid_python, METH_VARARGS,
     "Compute the approximate smallest enclosing ellipsoid for a set of "
     "points."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef bounding_ellipsoid_module = {
    PyModuleDef_HEAD_INIT, "bounding_ellipsoid",
    "Compute the approximate smallest enclosing ellipsoid for a set of points.",
    -1, BoundingEllipsoidMethods};

PyMODINIT_FUNC PyInit_bounding_ellipsoid(void) {
    import_array();
    return PyModule_Create(&bounding_ellipsoid_module);
}
