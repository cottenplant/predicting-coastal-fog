FROM python:3.6-slim
MAINTAINER "Sam Cotten"
WORKDIR /app
ADD . /app
RUN pip install -r json urllib scikit-learn
ENV NAME Fog
CMD ["python", "app.py"]

