# file that defines a Docker image
FROM continuumio/anaconda3:2021.11
# might be wrong below
ADD . /code
WORKDIR /code
ENTRYPOINT ["python", "deployment/drug_app.py"]
# might need to take the files out of /deployment and put in CWD