from setuptools import setup, find_packages

setup(
    name='SampleHybridPriorWavletFlow',  # Replace with your package name
    version='0.1.0',
    author='Mati',
    author_email='your.email@example.com',
    description='sampling package for Hybrid Prior Wavelet-Flow',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your-repo-name',  # Update with your repository URL
    packages=find_packages(),  # Automatically find package directories
    install_requires=[
        # List your package dependencies here, e.g.:
        # 'numpy', 'pandas',
        'lmdb==1.4.1',
        'matplotlib==3.8.0',
        'numpy==1.24.1',
        'pandas==2.2.2',
        'pytorch-wavelets==1.3.0',
        'quantimpy==0.4.6',
        'scikit-image==0.23.2',
        'scipy==1.11.3',
        'seaborn==0.13.2',
        'torch==2.0.1',
        'torchaudio==2.0.2',
        'torchvision==0.15.2',
        'nflows',
        'PyWavelets',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10, <3.11',  # Specify the Python version requirement
)