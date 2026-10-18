# Lab 5: Containerizing ML Models with Docker

In this lab, you will containerize a machine learning training pipeline and an inference server using Docker. You will train a Wine classifier, serve it through a Flask API, and run both containers together with Docker Compose. The focus is **Docker volume management**: you will use a named volume to share the trained model between containers, and a bind mount to write prediction logs onto your own machine.

To receive credit for this lab, you must demonstrate your completed work to the TA during recitation.

## AI Usage Policy

This is an individual graded assignment. AI assistants may be used only for conceptual clarification and debugging guidance. AI assistants should not generate or complete the implementation, modify the starter code, or provide complete solutions to any TODO sections. You are responsible for understanding and writing your own implementation. During the lab demonstration, you may be asked to explain your code and make a small modification to verify your understanding.

Verification: You must be able to explain your implementation and make requested modifications during the TA demonstration. Work that cannot be explained or modified by the student may not receive credit.

## Deliverables

- [ ] **Deliverable 1**: Run the training script inside a container and save the resulting model to a shared volume. Explain why Docker is useful for reproducibility and portability in ML training.
- [ ] **Deliverable 2**: Containerize the inference service, serve predictions on a port, and show the TA the `./logs/predictions.log` file **on your host machine**. Explain what the Dockerfile does and how it containerizes the service.
- [ ] **Deliverable 3**: Call the health endpoint before and after destroying the named volume to show how model availability changes. Explain the difference between a named volume and a bind mount.

## Getting started

Clone the starter code from this Git repository:

```
git clone https://github.com/Srivatsa03/mlip-docker-lab.git
cd mlip-docker-lab
```

The repository includes:
- `docker/training/` — `train.py`, `requirements.txt` and a partially complete `Dockerfile`
- `docker/inference/` — `server.py` (you will write the Dockerfile and requirements yourself)
- `docker-compose.yml` — a skeleton you will complete in Step 3

**Work from a directory your Docker installation is allowed to share with your host**, such as somewhere under your home folder. Docker Desktop shares `/Users` by default, and Colima shares your home directory. If you clone into a path your Docker VM does not share, the bind mount in Step 2 will silently write the log file inside the container only and nothing will appear on your host, which is Deliverable 2.

## Step 0: Setup Docker

Install Docker on your system and verify the installation:

```bash
docker run hello-world
```

Follow the instructions for your operating system: https://docs.docker.com/get-docker/

## Step 1: Containerize the Training Pipeline

Create a Docker container for the training code. When launched, the container should train a Wine classifier and save the model file to a shared volume. Use the partially completed Dockerfile and code in `docker/training/`.

### 1a. Complete `train.py`

Create a `RandomForestClassifier` and train it on the Wine dataset, then save the trained model with `joblib.dump()` to `/app/models/wine_model.pkl`.

### 1b. Complete the Training Dockerfile

Open `docker/training/Dockerfile`. It is nearly complete. Fill in the TODO that sets the command which runs the training script.

### 1c. Build and Run with a Named Volume

Build the training image and run it with a **named volume** mounted for model storage:

```bash
docker build -t mlip-training -f docker/training/Dockerfile .
docker run --rm -v wine_model_storage:/app/models mlip-training
```

You should see the test accuracy and a message that the model was saved. On this dataset and split the accuracy is high, so do not be surprised by a number close to 1.0.

**What is a named volume?** With `-v wine_model_storage:/app/models`, Docker creates and manages a volume called `wine_model_storage`. Its data survives after the container exits.

## Step 2: Containerize the Inference Server

Create a container that loads the trained model from the shared volume and serves predictions through a Flask API. The server also logs each prediction to a bind-mounted directory so you can read it from your host.

### 2a. Complete `server.py`

Open `docker/inference/server.py`. You need to load the trained model from the shared volume, pull the features out of the incoming JSON request, run the prediction, and append a line for each prediction to `/app/logs/predictions.log`.

The server already has a `/health` endpoint that reports whether the model file exists, which is useful when debugging volume problems.

Think about what should happen when the model file is not there yet. The model is loaded once when the process starts, so whatever you do at that moment decides what `/health` reports later. You will test exactly that case in Step 4.

### 2b. Create the Inference Dockerfile

Create `docker/inference/Dockerfile` from scratch. It should:

- Use `python:3.11-slim` as the base image
- Set the working directory to `/app`
- Copy and install dependencies from a requirements file
- Copy `server.py` into the working directory
- Expose the port the Flask app listens on
- Set the command that runs `server.py`

**HINT**: Look at the training Dockerfile for reference. The build context is the project root, so paths are written as `docker/inference/...`.

Check which port `server.py` actually binds before you write the `EXPOSE` line. Inside the container the app listens on **8080**, and you will publish it on host port 8081 in the next step. Exposing a port that nothing listens on is a common and confusing mistake.

You also need to create `docker/inference/requirements.txt` with Flask, scikit-learn, joblib and numpy. **Pin scikit-learn to the same version as `docker/training/requirements.txt`.** A model pickled by one version and loaded by another can warn or fail outright.

### 2c. Build and Run with Both Volume Types

Create a local directory for logs, then run the inference container with a named volume for the model and a bind mount for the logs:

```bash
mkdir -p ./logs
docker build -t mlip-inference -f docker/inference/Dockerfile .
docker run --rm -p 8081:8080 \
  -v wine_model_storage:/app/models \
  -v $(pwd)/logs:/app/logs \
  mlip-inference
```

The two `-v` flags do different things:
- `wine_model_storage:/app/models` is a **named volume**, managed by Docker and shared with training
- `$(pwd)/logs:/app/logs` is a **bind mount**, which maps your own `./logs/` directory into the container

### 2d. Test the Inference Server

Check the health endpoint:

```bash
curl http://localhost:8081/health
```

Send a prediction request with the 13 Wine features:

```bash
curl -X POST http://localhost:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"input": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050]}'
```

Test the error path with a bad request:

```bash
curl -X POST http://localhost:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"bad_key": [1,2,3]}'
```

Then look at your local `./logs/` directory. A `predictions.log` file with timestamped entries should be there. That is the bind mount at work: the container writes to `/app/logs/` and the file lands on your host filesystem. If the file shows up inside the container but not on your host, re-read the note in Getting started about shared paths.

## Step 3: Docker Compose

Docker Compose lets you define a multi-container application instead of typing long command lines. Complete `docker-compose.yml` for both services.

You need to fill in:
- The build context and Dockerfile path for each service
- The **named volume** `wine_model_storage` mounted at `/app/models` on both services
- The **bind mount** from `./logs` to `/app/logs` on the inference service
- The port mapping for the inference service
- The named volume definition in the `volumes:` section at the bottom

Then run:

```bash
docker compose up --build
```

Two things to watch for here, because both will bite you:

**Training has to finish before inference loads the model.** A plain `depends_on: - training` only waits for the training container to *start*, not to finish, so the server can look for the model while training is still writing it. Use the form that waits for a clean exit:

```yaml
depends_on:
  training:
    condition: service_completed_successfully
```

**Compose renames your volume.** Compose prefixes volume names with the project name, so the volume from Step 1c and the one Compose creates are two different volumes. Run `docker volume ls` and you will see both. If you want Compose to use the same unprefixed name, declare it:

```yaml
volumes:
  wine_model_storage:
    name: wine_model_storage
```

Once both services are up, test with the same curl commands from Step 2d and confirm that predictions return a wine class and that `./logs/predictions.log` on your host is being written to.

Shut it down:

```bash
docker compose down
```

## Step 4: Volume Lifecycle

This step shows how Docker handles persistence and host sharing differently for each mount type.

### 4a. Inspect the Named Volume

```bash
docker volume ls
docker volume inspect <the volume name you see in the list>
```

Look at the `Mountpoint` field. That is the Docker-managed path holding your model. On Mac and Windows that path lives inside the Docker VM, not on your own filesystem, which is part of the point.

### 4b. Persistence Check

Start both services:

```bash
docker compose up --build
```

Confirm health with `curl http://localhost:8081/health`.

Stop and remove the containers, keeping the volume:

```bash
docker compose down
```

Bring back only the inference service:

```bash
docker compose up --no-deps inference
```

Check health again. It should still report healthy, which shows the model survived in the named volume even though both containers were destroyed.

### 4c. Removing the Volume

Now wipe the volume as well:

```bash
docker compose down -v
```

Start only the inference service again, with `--no-deps` so that training does not simply recreate the model:

```bash
docker compose up --no-deps inference
```

Check health one more time. Be ready to tell the TA what you see, why it differs from 4b, and what your choice in Step 2a had to do with it. Depending on how you handled a missing model file, the service either reports that the model is missing or never stays up at all. Both are defensible, and knowing which one you built is the point.

## Additional Resources

1. [Docker For Beginners](https://docker-curriculum.com/)
2. [Docker Volumes Documentation](https://docs.docker.com/storage/volumes/)
3. [Docker Bind Mounts Documentation](https://docs.docker.com/storage/bind-mounts/)

## Troubleshooting

- Check that the Docker daemon is running
- Check that port 8081 is free on your host
- Read service logs with `docker compose logs`
- Make sure training finished before inference tried to load the model
- Use `docker compose exec` to look inside a running container's filesystem
- If the model file is missing, confirm the volume is mounted with `docker volume ls` and remember the Compose name prefix
- If logs appear in the container but not on your host, your project directory is probably outside the paths Docker shares with your machine
- If `joblib.load` complains about versions, your two requirements files disagree on scikit-learn
