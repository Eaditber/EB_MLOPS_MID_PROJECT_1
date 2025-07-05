from setuptools import setup, find_packages

setup(
    name='myml_package',  # This is the name you'll `pip install`
    version='0.1.0',
    packages=find_packages(), # Automatically finds your 'my_module'
    description='A custom Python package for Airflow DAGs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Eadiit Bernstein',
    author_email='eadit.bernstein@gmail.com',
    url='https://github.com/yourusername/my_custom_airflow_package', # Optional
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8', # Adjust based on your Airflow environment's Python version
    install_requires=[
        # List any external dependencies your package needs (e.g., 'pandas', 'requests')
    ],
)