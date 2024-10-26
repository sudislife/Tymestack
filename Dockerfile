FROM gcr.io/deeplearning-platform-release/base-cu110.py37

WORKDIR /

RUN pip install pandas
RUN pip install scikit-learn
RUN pip install cloudml-hypertune

# Copy the training code to the docker image.
COPY trainer /trainer

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
