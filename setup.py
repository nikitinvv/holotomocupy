from setuptools import setup, find_packages


setup(
    name='holotomocupy',
    version=open('VERSION').read().strip(),
    author='Viktor Nikitin',
    author_email='vnikitin@anl.gov',
    url='https://github.com/nikitinvv/holotomocupy',
    package_dir={"": "src"},
    packages=find_packages('src'),
    include_package_data=True,
    description='Framework for constructing advanced reconstruction method in holotomography',
    zip_safe=False,
)
