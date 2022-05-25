from setuptools import setup, find_packages

VERSION = '0.4.0'
DESCRIPTION = 'NGC-Learn'
LONG_DESCRIPTION = 'A toolkit library for building arbitrary predictive processing/coding architectures based on the neural generative coding (NGC) computational framework (Ororbia & Kifer 2022).'

packages = find_packages()
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'ngclearn' or p.startswith('ngclearn.')

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="ngclearn",
        version=VERSION,
        author="Alexander Ororbia",
        author_email="<ago@cs.rit.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=packages, #['gym_minigrid', 'gym_minigrid.envs'],
        url='https://github.com/ago109/ngc-learn',
        install_requires=[],
        #install_requires=['numpy>=1.20.0','tensorflow-gpu>=2.0.0'], # add any additional packages that'1.20.0'
        # needs to be installed along with your package

        keywords=['python', 'ngc-learn', 'predictive-processing', 'predictive-coding'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Research/Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Ubuntu",
        ]
)
