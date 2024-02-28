from setuptools import setup, find_packages

setup(
    name='qirnn',
    version='1.0',
    author='David Condrey',
    author_email='davidcondrey@protonmail.com',
    description='Quantum-Inspired Recursive Neural Network',
    long_description='A novel architecture inspired by principles from quantum computing and neural networks.',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/qirnn',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
