FROM python:3.12-slim
RUN useradd -m app
USER app
WORKDIR /home/app
COPY --chown=app server.py /home/app/server.py
ENV PORT=8080
CMD ["python", "/home/app/server.py"]
