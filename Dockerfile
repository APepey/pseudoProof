# OPTIMIZED SOLUTION
# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
# FROM tensorflow/tensorflow:2.10.0
FROM python:3.10-buster
# This line sets the working directory within the container to /prod.
# This directory will be used as the default location for subsequent commands.f
# libraries required by OpenCV
RUN apt-get update
# RUN apt-get install \
#   'ffmpeg'\
#   'libsm6'\
#   'libxext6'  -y
# We strip the requirements from useless packages for user
# This line copies the contents of the requirements_prod.txt file from the host machine
# to the /prod directory in the container  and renames it to requirements.txt.
COPY requirements.txt requirements.txt
# Update pip
RUN pip install --upgrade pip
# This line installs the Python dependencies listed in the requirements.txt file using the pip package manager.
# It will run the file which was originally requirementsAPI.txt
# These lines copy the directory, setup.py file, and any associated content from the host machine
# to the /prod directory in the container.
COPY setup.py setup.py
# This line installs the Python package located in the current directory (.),
# which includes the contents of the face_tally directory and is specified by the setup.py file.
RUN pip install .
# This line copies the Makefile from the host machine to the /prod directory in the container.
COPY Makefile Makefile
# This line runs the make reset_local_files command inside the container.
# It likely executes some tasks defined in the Makefile.
# (LINE COMMENTED UNTIL USEFULL MAKE COMMAND TO RUN)
# RUN make reset_local_files
# This line specifies the default command to run when the container starts.
# It uses uvicorn to run the FastAPI application (taxifare.api.fast:app) with specific host and port settings.
# The $PORT environment variable is expected to be provided at runtime.
COPY pseudoproof pseudoproof/
CMD uvicorn pseudoproof.fast.api:app --host 0.0.0.0 --port $PORT
# IMPORTANT!! Try before if it works of the console with this line:
# uvicorn face_tally.API.fast:app --reload --port 8000
###############################################
# SINTAX OF COPY:
# COPY <src> <dest>
# <src> represents the source path on the host machine (the machine where you're building the Docker image).
# <dest> represents the destination path inside the container.
# <src> is requirementsAPI.txt on the host machine.
# <dest> is requirements.txt inside the container.
# RUN vs CMD:
# The RUN instruction is used to execute commands during the build process of the Docker image.
# The CMD instruction is used to provide default command(s) that will be executed when a container is run from the built image.

# FROM python:3.10-buster

# RUN apt-get update

# COPY requirements.txt requirements.txt

# RUN pip install --upgrade pip

# RUN pip install -r requirements.txt

# COPY pseudoproof pseudoproof

# COPY setup.py setup.py

# RUN pip install .

# COPY Makefile Makefile

# CMD uvicorn pseudoproof.fast:app --host 0.0.0.0 --port $PORT
