from glob import glob

from setuptools import find_packages, setup

package_name = 'tabletop_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ananya Acharya',
    maintainer_email='aa2334@g.rit.edu',
    description='Tabletop object perception and antipodal grasp detection (ROS-free core + node)',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = tabletop_perception.perception_node:main',
        ],
    },
)
