from setuptools import setup, Extension


module = Extension(
    "bounding_ellipsoid",
    sources=["bounding_ellipsoid_bindings.cpp"],
)

setup(
    name="bounding_ellipsoid",
    version="1.0",
    description="Bindings for CGAL to compute a minimum bounding ellipsoid.",
    author="Adam Heins",
    install_requires=["numpy"],
    ext_modules=[module],
    zip_safe=False,
)
