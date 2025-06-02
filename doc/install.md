# PMosquito – Assembly of interrupted trajectory data by videotracking

## How to install 

## Installation of Pmosquito

### Open a terminal
A terminal (or command prompt) is a tool that allows you to interact with your computer by typing text commands. Unlike a graphical interface (where you click buttons), the terminal allows you to execute specific instructions, such as launching a Python script, installing libraries, or navigating through your project folders.

**Windows**: Press Windows + R, type cmd, and then press Enter.

**macOS**: Open the Terminal application via Spotlight (Cmd + Space, then type "Terminal").

**Linux**: Use the shortcut Ctrl + Alt + T or search for "Terminal" in your applications menu.

To install Pmosquito, you need Python 

### Check if Python is Installed

Type the following command:

	python --version

or 

	python3 --version

If you see something like Python 3.x.x, Python is already installed. Otherwise, proceed to the next step.

### Install Python

**For Windows**

Go to https://www.python.org/downloads/windows/

Click Download Python 3.x.x

IMPORTANT: On the first installation screen, check the box "Add Python to PATH"

Click Install Now

After installation, reopen your terminal and check:

	python --version

**For macOS**

If needed, install Homebrew

Use the following command:

	brew install python


**For Linux (Debian, Ubuntu, etc.)**

Use the following commands:

	sudo apt update
	sudo apt install python3 python3-venv python3-pip

### Install pip (if not already installed)

Most modern Python installations include pip. To check:

	pip --version

or

	pip3 --version

If it's missing:

On Windows, reinstall Python and ensure "Install pip" is checked

On Linux, use:

	sudo apt install python3-pip

### Virtual Environment

It is recommended to use a virtual environment:

- Each virtual environment contains its own version of Python and its own packages, independent of other projects or the system installation. This prevents a package update for one project from breaking another.
- You can install exactly the versions of libraries needed for one project without affecting others. Perfect for replicating an environment on another machine.

In your terminal you need to move to the Pmosquito-main directory
You must write in your terminal: 
	
 	cd path_name

**Note**: To find your path:
Open File Explorer.
Go to your project folder.
Click in the address bar at the top: the path will appear.

Then, type :

	python -m venv env

Here, env is the name of the virtual environment. You can choose a different name if you wish.
This creates an env/ folder containing an isolated Python installation.

Then ypu need to activate your virtual environment. Type always in your terminal :

**For Windows**

	.\env\Scripts\activate

**For Linux/macOS**

	source env/bin/activate

Normally you should see your virtual environment name displayed in parentheses at the beginning of the new command line

### Retrieve this file containing the codes and test 

[Download PMosquito as a ZIP file](https://github.com/AnaisDenis/PMosquito/archive/refs/heads/main.zip)

You can extract this folder to the directory of your choice: you will find the .zip file in your downloads. Move it by right-clicking, cutting, and pasting it into the directory of your choice. Then right-click and extract here.

### Install Pmosquito

Once your virtual environment is activated and you are in the project directory (the folder that contains `pyproject.toml`), install Pmosquito and its dependencies with:

    pip install .

This command will install:
- the `Pmosquito` tool,
- all required dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

