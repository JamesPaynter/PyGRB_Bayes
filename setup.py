import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='PyGRB_Bayes-jpaynter',
    version='0.0.1',
    author='James Paynter',
    author_email='jpaynter@student.unimelb.edu.au',
    description='Opens GRB FITS files, fits pulses to light-curves using Bilby.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/JamesPaynter/PyGRB_Bayes',
    packages=setuptools.find_packages(),
    package_dir = {'PyGRB_Bayes' : 'PyGRB_Bayes'},
    package_data={'PyGRB_Bayes': ['data/*']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7.0',
)
