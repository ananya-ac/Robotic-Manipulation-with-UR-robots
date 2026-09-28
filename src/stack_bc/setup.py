from setuptools import setup

package_name = 'stack_bc'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ananya Acharya',
    maintainer_email='aa2334@g.rit.edu',
    description='Cube-stacking behavior-cloning data pipeline for the MuJoCo sim',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'build_scene = stack_bc.scene:main',
            'feasibility = stack_bc.feasibility:main',
        ],
    },
)
