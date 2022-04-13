CUSTOMER CHURN PREDICTION

About
A system for helping companies retain their customers and thus reduce the churn rate created using Machine Learning.


Installation commands:

How to run locally:

The following instructions will enable you to set up and run the project locally for development and testing.
Pre-requisites:
Create a virtual environment using Anaconda.

The following Python libraries have been used:
Flask, Numpy, Pandas, sklearn, seaborn which will be installed during setting the environment.
After the requirements are installed in your system, you can simply execute the .py script named app.py.
For deployment purposes it is essential to create a virtual environment.

Step 1: Main Directory
Go to command prompt and change to the project as the default directory. 
Example: C:\Users\name>cd Desktop\flasksite

Step 2: Create environment
Create a conda environment using 'conda create -n env'.
This will create an Python environment with name env. We can activate using ‘activate env’.
virtualenv venv
venv\scripts\activate
To exit the virtual environment, we can execute 'deactivate env'.

Step 3: Install required libraries
There are default libraries in the Anaconda distribution. 
Import all libraries imported in flaskapp.py! 
We can use pip - Python default package managerpip install [nameOfPackage] to install the libraries specified in pre-requisites Example:
pip install flask
pip install numpy
pip install seaborn
pip install sklearn

Step 4: Execute script
When all python dependencies are installed, execute the .py script named app.py or using command flask run in the command prompt.
You should see your flask code up and running after you follow the link!

