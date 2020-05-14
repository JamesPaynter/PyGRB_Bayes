# NEED THIS INIT FOR PYTEST --COV TO GENERATE A PROPER REPORT


# Found the answer:
#
# DO NOT put a __init__.py file in a folder containing TESTS if
# you plan on using pytest. I had one such file, deleting it
# solved the problem.
#
# This was actually buried in the comments to the second
# answer of PATH issue with pytest
# 'ImportError: No module named YadaYadaYada' so I did
# not see it, hope it gets more visibility here.

#https://stackoverflow.com/questions/41748464/pytest-cannot-import-module-while-python-can
