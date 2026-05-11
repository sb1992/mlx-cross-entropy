from setuptools import setup
from mlx import extension

if __name__ == "__main__":
    setup(
        name="mlx-cross-entropy",
        version="0.1.0",
        description="Memory-efficient Cut Cross-Entropy for MLX with native Metal kernels.",
        ext_modules=[extension.CMakeExtension("mlx_cce._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mlx_cce"],
        package_data={"mlx_cce": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.9",
    )
