# Conditional flow matching with PyTorch Lightning

Create a virtual environment and install required packages:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train a flow:
```
nohup python -u train.py &> output.out &
```

During or after training, check metrics:
```
tensorboard --logdir=logs &
```
