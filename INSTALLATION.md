# Python Installation Guide & PMosquito Setup

This guide walks you through:

- Installing Python (Windows, macOS, Linux)
- Installing `pip`
- Creating a virtual environment

---

## Check if Python is Installed

Open your terminal:

- **Windows**: PowerShell
- **macOS/Linux**: Terminal

Type the following command:

	python --version

or 

	python3 --version

If you see something like Python 3.x.x, Python is already installed. Otherwise, proceed to the next step.

## Install Python

### For Windows

Go to https://www.python.org/downloads/windows/

Click Download Python 3.x.x

IMPORTANT: On the first installation screen, check the box "Add Python to PATH"

Click Install Now

After installation, reopen your terminal and check:

	python --version

### For macOS

If needed, install Homebrew

Use the following command:

	brew install python


### For Linux (Debian, Ubuntu, etc.)

Use the following commands:

	sudo apt update
	sudo apt install python3 python3-venv python3-pip

## Install pip (if not already installed)

Most modern Python installations include pip. To check:

	pip --version

or

	pip3 --version

If it's missing:

On Windows, reinstall Python and ensure "Install pip" is checked

On Linux, use:

	sudo apt install python3-pip

## Virtual Environment

It is recommended to use a virtual environment:

- Chaque environnement virtuel contient sa propre version de Python et ses propres packages, indépendamment des autres projets ou de l'installation système. Cela évite qu'une mise à jour d’un package pour un projet n’en casse un autre.
- Vous pouvez installer exactement les versions nécessaires des bibliothèques pour un projet sans affecter les autres. Parfait pour reproduire un environnement sur une autre machine.


Pour ce faire, dans votre terminal (ou invite de commandes), placez-vous dans le dossier de votre projet, puis tapez :

	python -m venv env

Ici, env est le nom de l’environnement virtuel. Vous pouvez choisir un autre nom si vous le souhaitez.
Cela crée un dossier env/ contenant une installation isolée de Python.

### For Windows

	.\env\Scripts\activate

### For Linux/macOS

	source env/bin/activate


