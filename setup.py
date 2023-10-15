from setuptools import setup, find_packages

VERSION = '0.6.0'
DESCRIPTION = 'NGC-Learn'
LONG_DESCRIPTION = 'Simulation software for building and analyzing arbitrary predictive coding, spiking network, and biomimetic neural systems.'

packages = find_packages()
# Ensure that we do not pollute the global namespace.
for p in packages:
    assert p == 'ngclearn' or p.startswith('ngclearn.')

# Setting up software package construction
setup(
       # name of package must match the folder name 'ngclearn'
        name="ngclearn",
        version=VERSION,
        author="Alexander Ororbia",
        author_email="<ago@cs.rit.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=packages,
        license="BSD-3-Clause License",
        url='https://github.com/ago109/ngc-learn',
        install_requires=[], # add any additional packages that need to be installed along with your package
        keywords=['python', 'ngc-learn', 'predictive-processing', 'predictive-coding',
                  'jax', 'spiking-neural-networks', 'biomimetics', 'bionics'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD-3-Clause License",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Cognitive Science",
            "Topic :: Scientific/Engineering :: Computational Neuroscience",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: Linux :: Ubuntu"
        ]
)
