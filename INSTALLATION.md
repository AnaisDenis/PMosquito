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

If you see something like Python 3.10.x, Python is already installed. Otherwise, proceed to the next step.

## Install Python
For Windows

Go to https://www.python.org/downloads/windows/

Click Download Python 3.x.x

IMPORTANT: On the first installation screen, check the box "Add Python to PATH"

Click Install Now

After installation, reopen your terminal and check:

	python --version

For macOS

If needed, install Homebrew

Use the following command:

	brew install python


For Linux (Debian, Ubuntu, etc.)

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
