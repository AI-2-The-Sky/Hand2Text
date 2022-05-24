To launch the docker
```bash
docker run -v ~/42/42-AI/Hand2Text/:/home/Hand2Text -it --rm ubuntu /bin/bash
```

To execute the test locally
```bash
cd /home/Hand2Text/
export CI=TRUE
apt-get update && apt-get install python3 python3-pip ffmpeg libsm6 libxext6  -y
.42AI/init.sh /usr/bin/python3
#  pre-commit run -a
ln -s /usr/bin/python3 /usr/bin/python
export HYDRA_FULL_ERROR=1
python3 -m pytest -c tests/shell/test_sweeps.py -k "test_optuna_sweep"
```
