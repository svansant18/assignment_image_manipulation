# Numpy assignment: Image manipulation
The codesource in this project will be able to render the images from the assignment.pdf file found under the documentation folder.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Before starting with the project, make sure you have **Anaconda** installed, which will automatically install Python, and the **Jupyter Notebook**. This will ensure a smooth and hassle-free experience while running the code present in the repository. If you don’t have Anaconda installed, you can download it from the official website. After installation, you can start the Jupyter Notebook via the Anaconda Navigator or the command line.

### Installing <br>

##### 1. Downloading the GIT repository.<br>
To download the source code from this GitHub repository, you have two options:

##### Using Git
If you have Git installed on your system, open your terminal/command prompt and run the following command:

> git clone https://github.com/username/repository.git

*Replace username with the GitHub **username** and **repository** with the name of this repository.*

##### Downloading as ZIP
If you don’t have Git installed or prefer to download the code as a ZIP file, simply click on the green Code button on the repository page and then click *Download ZIP*. Once downloaded, extract the ZIP file to access the source code.

Remember to replace the URL in the git clone command with the actual URL of this repository.

<br>

##### 2. Setting up the environment.

Open the **Anaconda Powershell Prompt** navigate to the repository and create a new environment by using the *conda env create* command and specifying the path to the environment.yaml file in this project. This will create a new environment with the name and dependencies specified in the *environment.yaml* file.

>   *conda env create -f environment.yaml*

Check if the environment has been succesfully installed by using the command:

>  *conda info --envs*

You can then activate the environment using the conda activate command and the name of the environment. For example, in this project the default name is *assignment*, you can run:

>  *conda activate assignment*

You can now use the anaconda environment as usual. To deactivate the environment, you can run:

> *conda deactivate*

## Deployment
First, open the **Anaconda Navigator** application. You’ll see a list of applications available in your Anaconda environment. Click on the ‘Launch’ button under **Jupyter Notebook**. This will open a new tab in your web browser with a file explorer.

Navigate to the */notebook* directory containing assignment.pynb and click on it to open the notebook. Select a cell and press the ‘Run’ button or Shift + Enter to execute it.

 
## Built With
* VS Code
* Jupyter Notebook

## Author(s)
* **Steve Van Sant**

## Contributing
N/A

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
