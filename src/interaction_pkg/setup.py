from setuptools import find_packages, setup

package_name = 'interaction_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         ['launch/interaction.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='swarghane',
    maintainer_email='sourabhwarghane@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stt_node=interaction_pkg.stt_node:main',
            'tts_node=interaction_pkg.tts_node:main',
            'llm_node=interaction_pkg.llm_node:main',
        ],
    },
)
