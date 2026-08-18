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
    maintainer='Sourabh Warghane',
    maintainer_email='sourabhwarghane@gmail.com',
    description='ROS 2 human-robot interaction package providing wake-word detection, speech recognition, LLM-based interaction, vision queries, and text-to-speech.',
    license='Apache-2.0',
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
