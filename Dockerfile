FROM python:3.9-slim

EXPOSE 8501

WORKDIR /pyblood

COPY . .

RUN pip3 install -r requirements.txt


ENTRYPOINT ["streamlit", "run", "Introduction.py", "--server.port=8501", "--server.address=0.0.0.0"]