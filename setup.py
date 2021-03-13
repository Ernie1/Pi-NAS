import io
import os
import subprocess

from setuptools import setup, find_packages

cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.1.1'
try:
    if not os.getenv('RELEASE'):
        from datetime import date
        today = date.today()
        day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'encoding', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is encoding version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'portalocker',
    'torch>=1.4.0',
    'torchvision>=0.5.0',
    'Pillow',
    'scipy',
    'requests',
]

if __name__ == '__main__':
    create_version_file()
    setup(
        name="Pi-NAS",
        version=version,
        author="Anonymous-Pi-NAS",
        author_email="",
        url="https://github.com/Anonymous-Pi-NAS/Pi-NAS",
        description="Pi-NAS Package",
        long_description="",
        long_description_content_type='text/markdown',
        license='MIT',
        install_requires=requirements,
        packages=find_packages(exclude=["tests", "experiments"]),
        package_data={ 'encoding': [
            'LICENSE',
            'lib/cpu/*.h',
            'lib/cpu/*.cpp',
            'lib/gpu/*.h',
            'lib/gpu/*.cpp',
            'lib/gpu/*.cu',
        ]},
    )
