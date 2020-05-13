import setuptools

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='PyGRB-jpaynter',
    version='0.0.1',
    author='James Paynter',
    author_email='jpaynter@student.unimelb.edu.au',
    description='Opens GRB FITS files, fits pulses to light-curves using Bilby.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/JamesPaynter/PyGRB',
    packages=setuptools.find_packages(),
    package_dir = {'PyGRB' : 'PyGRB'},
    package_data={'PyGRB_Bayes': ['data/*']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7.0',
)
