FROM australia.gcr.io/deeplearning-platform-release/tensorflow-serving

RUN pip install --upgrade pip
RUN pip install cloudml-hypertune

# Copy the training code to the docker image.
COPY trainer /trainer

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]