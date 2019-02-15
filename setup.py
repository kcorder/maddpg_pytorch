from setuptools import setup, find_packages

setup(name='maddpg_pytorch',
      version='0.0.1',
      description='PyTorch Implementation of MADDPG',
      url='https://github.com/kcorder/pytorch_maddpg',
      author='Kevin Corder',
      author_email='kcorder@udel.edu',
      packages=find_packages(),
      include_package_data=True,
      # install_requires=['gym', 'numpy-stl']
)
