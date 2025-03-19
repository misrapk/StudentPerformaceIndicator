from setuptools import setup, find_packages
from typing import List
with open("README.md", "r") as fh:
    description = fh.read()
    
def get_requirements(filePath:str) -> List[str]:
    """Return the list of requirement"""
    requirements = []
    with open(filePath) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .")
            
    return requirements
            
    
setup(
    name="StudentPerformanceIndicator",
    version="0.1.0",
    author="Peeyush",
    author_email="pkmisra1999@gmail.com",
    description="A simple machine learning project to get the performance of a student",
    url="https://github.com/yourusername/my_ml_project",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.7',
)