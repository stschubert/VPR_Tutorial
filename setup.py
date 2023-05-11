import os, sys
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


install_require_list = [
    'numpy', 'matplotlib', 'Pillow', 'scipy',
    'scikit-image', 'tensorflow', 'tensorflow_hub',
    'torch', 'torchvision', 'tqdm']

# workaround as opencv-python does not show up in "pip list" within a conda environment
# we do not care as conda recipe has py-opencv requirement anyhow
is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
if not is_conda:
    install_require_list.append('opencv-python')

setup(name='vpr_tutorial',
      version='1.0.0',
      description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.',
      long_description = long_description,
      long_description_content_type='text/markdown',
      author='Stefan Schubert, Peer Neubert, Sourav Garg, Michael Milford and Tobias Fischer',
      author_email='stefan.schubert@etit.tu-chemnitz.de',
      url='https://github.com/QVPR/Patch-NetVLAD',
      license='GPL-3.0-or-later',
      classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
      ],
      python_requires='>=3.8',
      install_requires=install_require_list,
      packages=find_packages(),
      keywords=[
          'python', 'place recognition', 'image retrieval', 'computer vision', 'robotics'
      ],
      scripts=['demo.py'],
      entry_points={
        'console_scripts': ['vpr-tutorial-demo=demo:main',],
      },
    #   package_data={'': ['configs/*.ini', 'dataset_gt_files/*.npz', 'example_images/*',
    #                      'output_features/.hidden', 'pretrained_models/.hidden', 'results/.hidden',
    #                      'dataset_imagenames/*.txt']}
)
