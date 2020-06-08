# Getting set up

In this course, we will primarily using Python as our primary programming language. We will also encourage participants to use R and Julia, but these will be unsupported, so use at your own risk. The code will be presented using Jupyter notebooks. The course will be hosted on github, and I encourage everyone to use git to pull from the repository. 

I will provide installation instructions for Windows and Mac users. If you are using Linux, I can trust you can figure it out.


## Anaconda (including Python)

1. Follow the installation instructions on the [Anaconda website](https://docs.anaconda.com/anaconda/install/), and download the Python 3.7 version.

2. For commands in `conda` Mac users can just use terminal, Windows users should open `Anaconda Prompt`.

3. Open Jupyter notebook using Anaconda Navigator or by typing 

   ```
   jupyter notebook
   ```

   in conda and open an `Python 3` notebook.

## Git

1. Install git
   * **Mac users:** open terminal and run ```git --version```. If you don't have git installed, you will be prompted to do so.
   * **Windows users:** go to the [Git for Windows](https://gitforwindows.org/) download page and install. I recommend also installing [Visual Studio Code](https://code.visualstudio.com/) for a text editor, unless you already use SublimeText or vim. Alternatively [GitHub Desktop](https://desktop.github.com/) is a nice GUI for Git.

2. Open git

3. Navigate to your home directory

   * **Mac:**`cd /Users/user/my_project`
   * **Windows:** `cd /c/user/my_project`

4. Clone the mec_optim repo
   ```
     git clone https://github.com/math-econ-code/mec_optim_2020-01
   ```
5. Whenever the repository is updated 

   ```
     git pull ori master 
   ```
   This is only scratching the surface of what we can do with Git and GitHub. It is an amazing way to version control and collaborate on your code. There are lots of great tutorials on how to use both Git and GitHub and I strongly recommend you get into the habit of using it.

## Docker (optional)

1. Install Docker
   * Mac and Windows users can install Docker Desktop from https://www.docker.com/products/docker-desktop. 
   * If your Windows is the Pro edition, you should be fine. 
   * If your Windows edition is the Home edition, you may run into problems. Try to install Windows subystem for Linux (https://docs.microsoft.com/en-us/windows/wsl/install-win10). This may take you to upgrade Windows.
   
2. Download the m-e-c.Dockerfile.
 
3. To build the container: Open the shell, and cd to the directory where the dockerfile is, and run
   ```
     docker build --tag=m-e-c:latest --tag=m-e-c:stable -f m-e-c.Dockerfile .
   ```
   (it may take a while).
4. To run the container: create a local folder, and run
   ```
     docker run -it -p 8888:8888 -v <your/local/folder>:/home/mec m-e-c
   ```
5. To launch the Jupyter notebook 
   ```
     cd  ../home/mec
     jupyter lab --ip=0.0.0.0 --allow-root
   ```
   then open a browser and go to the URL as prompted. 


