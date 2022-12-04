#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install pytorch-ignite')


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics.metric import reinit__is_reduced


output_dir = Path(f'/content/drive/My Drive/CERT_OUTPUT')
answers_dir = Path(f"/content/drive/My Drive/answers")
main_answers_file = answers_dir / "insiders.csv"
log_dir = output_dir / 'logs'
checkpoint_dir = output_dir / 'checkpoints'

processed_data = "./merged.pkl"
dataset_version='5.2'

assert(output_dir.is_dir())
assert(answers_dir.is_dir())


# In[ ]:


class LSTM_Encoder(nn.Module):
	def __init__(self, padding_idx=None):
		super(LSTM_Encoder, self).__init__()

		self.input_size = 64

		self.embedding = nn.Embedding(64, 40, padding_idx=padding_idx)
		lstm_input_size = 40

		self.one_hot_encoder = F.one_hot
		
		self.lstm_encoder = nn.LSTM(
			lstm_input_size,
			40,
			num_layers=3,
			dropout=0.5,
			batch_first=True)
		self.dropout = nn.Dropout(0.5)
		self.decoder = nn.Linear(40,64)
		self.log_softmax = nn.LogSoftmax(dim=2)

	def forward(self, sequence):
		if self.embedding:
			x = self.embedding(sequence)
		else:
			x = self.one_hot_encoder(sequence,
				num_classes=self.input_size).float()
		x, _ = self.lstm_encoder(x)

		if self.training:
			x = self.dropout(x)
			x = self.decoder(x)
			x = self.log_softmax(x)
			return x
		else:
			return x
		

class CNN_Classifier(nn.Module):
	def __init__(self):
		super(CNN_Classifier, self).__init__()

		self.seq_length = 200
		self.lstm_hidden_size = 40

		self.conv1 = nn.Conv2d(1,32,kernel_size=5,padding=2)
		self.maxpool1 = nn.MaxPool2d(2, stride=2)
		self.conv2 = nn.Conv2d(32,64,kernel_size=5,padding=2)
		self.maxpool2 = nn.MaxPool2d(2, stride=2)

		self.flatten = lambda x: x.view(x.size(0),-1)
		self.linear = nn.Linear(64 * 200 * 40 // 16,2)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		assert(x.shape[2] == self.seq_length)
		assert(x.shape[3] == self.lstm_hidden_size)
		x = self.conv1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.maxpool2(x)
		x = self.flatten(x)
		x = self.linear(x)
		return x

class InsiderClassifier(nn.Module):
	def __init__(self, lstm_checkpoint):
		super(InsiderClassifier, self).__init__()
		self.lstm_encoder = LSTM_Encoder()
		self.lstm_encoder.requires_grad = False
		self.lstm_encoder.eval()
		self.load_encoder(lstm_checkpoint)

		self.sigmoid = nn.Sigmoid()
		self.cnn_classifier = CNN_Classifier()

	def train(self, mode=True):
		self.training = mode
		self.sigmoid.train(mode)
		self.cnn_classifier.train(mode)
		return self

	def load_encoder(self, checkpoint, device='cuda'):
		self.lstm_encoder.load_state_dict(
			torch.load(
				checkpoint,
				map_location=torch.device(device)),
			strict=True
			)
		return self

	def forward(self, x):
		with torch.no_grad():
			hidden_state = self.lstm_encoder(x)
			hidden_state = self.sigmoid(hidden_state)
		scores = self.cnn_classifier(hidden_state[:,None])

		return scores


# In[ ]:


class CertDataset(Dataset):
	@staticmethod
	def prepare_dataset(pkl_file, answers_csv, min_length=50, max_length=200):

		df = pd.read_pickle(pkl_file)
		df = df.reset_index().dropna()

		main_df = pd.read_csv(answers_csv)
		main_df = main_df[main_df['dataset'].astype(str) == str(dataset_version)]			.drop(['dataset', 'details'], axis=1)

		main_df['start'] = pd.to_datetime(main_df['start'], format='%m/%d/%Y %H:%M:%S')
		main_df['end'] = pd.to_datetime(main_df['end'], format='%m/%d/%Y %H:%M:%S')

		df = df.merge(main_df, left_on='user', right_on='user', how='left')
		df['malicious'] = (df.day >= df.start) & (df.day <= df.end)
		df = df.drop(['start', 'end', 'day', 'user'], axis=1)

		df['action_length'] = df.action_id.apply(len)

		df = df[df.action_length < min_length]

		df['action_id'] = df.action_id.apply(lambda x: x[:max_length])
		df['action_id'] = df.action_id.apply(lambda x: x + [0] * (max_length - len(x)))

		x = np.vstack(df.action_id.values)
		y = df.malicious.values

		return x, y

	def __init__(self, x, y, transform=None):
		self.x = x
		self.y = y.astype(int)
		self.transform = transform

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample = {'x': self.x[idx], 'y': self.y[idx]}

		if self.transform:
			sample = self.transform(sample)

		return sample

def get_dataset(data, ans):
    x, y = CertDataset.prepare_dataset(data, ans)
    return CertDataset(x, y)

def get_data_loaders(dataset, shuffle_dataset=True, validation_split=0.3, batch_size=1024, random_seed=0):
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))
	if shuffle_dataset:
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
											   sampler=train_sampler)
	validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
													sampler=valid_sampler)
	
	return train_loader, validation_loader


# In[ ]:


class Gen_accuracy(Accuracy):
	def __init__(self, ignored_class, *args, **kwargs):
		self.ignored_class = ignored_class
		super(Accuracy, self).__init__(*args, **kwargs)

	@reinit__is_reduced
	def update(self, output):
		y_pred, y = output

		indices = torch.argmax(y_pred, dim=1)

		mask = (y != self.ignored_class)
		mask &= (indices != self.ignored_class)
		y = y[mask]
		indices = indices[mask]
		correct = torch.eq(indices, y).view(-1)

		self._num_correct += torch.sum(correct).item()
		self._num_examples += correct.shape[0]


def prepare_batch(batch, device='cpu', train=True):
	x = batch['x']
	x = x.to(device).to(torch.int64)
	if train:
		y = x[:,1:]
		x = x[:,:-1]
		return x, y
	else:
		return x, batch['y']


def get_lstm_train_engine(model, optimizer, criterion, prepare_batch,
		device=None,
		log_dir=log_dir,
		checkpoint_dir=checkpoint_dir,
		checkpoint=None,
		tensorboard_every=10,
	) -> Engine:

	def _update(batch):
		model.train()
		optimizer.zero_grad()

		x, y = prepare_batch(batch, device=device)
		scores = model(x).transpose(1,2)

		loss = criterion(scores, y)
		loss.backward()
		optimizer.step()

		return {'loss': loss.item(), 'y_pred': scores, 'y': y}

	model.to(device)
	engine = Engine(_update)

	RunningAverage(output_transform=lambda x: x['loss']).attach(engine, 'average_loss')
	Accuracy().attach(engine, 'accuracy')
	Gen_accuracy(ignored_class=0).attach(engine, 'gen_accuracy')

	tb_logger = TensorboardLogger(log_dir=log_dir + '/train')
	tb_logger.attach(
		engine,
		log_handler=OutputHandler(
			tag="train",
            output_transform=lambda x: {"batch_loss": x['loss']},
            metric_names=['average_loss']),
        event_name=Events.ITERATION_COMPLETED(every=1))
	tb_logger.attach(
		engine,
		log_handler=OutputHandler(
			tag="train",
			output_transform=lambda x: {"epoch_loss": x['loss']},
            metric_names=['gen_accuracy', 'accuracy'],
			global_step_transform=global_step_from_engine(engine)),
		event_name=Events.EPOCH_COMPLETED,
	)

	tb_logger.attach(
		engine,
		log_handler=GradsScalarHandler(model, reduction=torch.norm, tag="grads"),
		event_name=Events.ITERATION_COMPLETED(every=tensorboard_every)
	)
	tb_logger.attach(
		engine,
		log_handler=GradsHistHandler(model, tag="grads"),
		event_name=Events.ITERATION_COMPLETED(every=tensorboard_every))

	to_save = {'model': model, 'optimizer': optimizer, 'engine': engine}
	checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpoint_dir, create_dir=True), n_saved=3)
	final_checkpoint_handler = Checkpoint(
		{'model': model},
		DiskSaver(checkpoint_dir, create_dir=True),
		n_saved=None,
		filename_prefix='final_'
	)

	if checkpoint:
		e = Events.ITERATION_COMPLETED(every=checkpoint)
	else:
		e = Events.EPOCH_COMPLETED
	engine.add_event_handler(e, checkpoint_handler)
	engine.add_event_handler(Events.COMPLETED, final_checkpoint_handler)

	@engine.on(Events.EPOCH_COMPLETED)
	def log_validation_results(engine):
		metrics = engine.state.metrics
		print(f"Epoch results - Avg loss: {metrics['average_loss']:.6f}, Accuracy: {metrics['accuracy']:.6f}, Non-Pad-Accuracy: {metrics['gen_accuracy']:.6f}")

	return engine


def get_lstm_validation_engine(
	model: torch.nn.Module,
	prepare_batch,
	criterion,
	device = None,
	non_blocking: bool = False,
	log_dir=log_dir,
	checkpoint_dir=checkpoint_dir,
) -> Engine:

	if device:
		model.to(device)

	def _inference(batch):
		model.train()
		with torch.no_grad():
			x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
			scores = model(x).transpose(1,2)
			return (scores, y)

	engine = Engine(_inference)

	Loss(criterion, output_transform=lambda x: x).attach(engine, 'epoch_loss')
	Accuracy().attach(engine, 'accuracy')
	Gen_accuracy(ignored_class=0).attach(engine, 'gen_accuracy')

	tb_logger = TensorboardLogger(log_dir=log_dir + '/validation')
	tb_logger.attach(engine,log_handler=OutputHandler(
			tag="validation",
			metric_names="all",
			global_step_transform=lambda x, y : engine.train_epoch),
            event_name=Events.EPOCH_COMPLETED)
	to_save = {'model': model}
	best_checkpoint_handler = Checkpoint(
		to_save,
		DiskSaver(checkpoint_dir, create_dir=True),
		n_saved=1, filename_prefix='best',
		score_function=lambda x: engine.state.metrics['gen_accuracy'],
		score_name="gen_accuracy",
		global_step_transform=lambda x, y : engine.train_epoch)
	engine.add_event_handler(Events.COMPLETED, best_checkpoint_handler)

	@engine.on(Events.COMPLETED)
	def log_validation_results(engine):
		metrics = engine.state.metrics
		print(f"Validation Results - ")
		for k in metrics.keys():
				print(k, ":", metrics[k])
	return engine


# In[ ]:


train_loader, val_loader = get_data_loaders(get_dataset(processed_data, main_answers_file))
device = 'cuda'


# In[ ]:


lstm_encoder = LSTM_Encoder()
criterion = nn.NLLLoss()
optimizer = optim.Adam(lstm_encoder.parameters())

train_engine = get_lstm_train_engine(lstm_encoder,
                                    optimizer, criterion, device=device,
                                    prepare_batch=prepare_batch)

val_engine = get_lstm_validation_engine(
        lstm_encoder, device=device,
        prepare_batch=prepare_batch,
        criterion=criterion)

@train_engine.on(Events.STARTED)
def log_training_results():
    print('Initial validation run:')
    val_engine.train_epoch = 0
    val_engine.run(val_loader)

@train_engine.on(Events.EPOCH_COMPLETED)
def log_training_results():
    print('Validation run:')
    val_engine.train_epoch = train_engine.state.epoch
    val_engine.run(val_loader)


# In[ ]:


train_engine.run(train_loader, max_epochs=500)


# In[ ]:


get_ipython().run_line_magic('ls', '"{checkpoint_dir}"')


# In[ ]:


# from google.colab import files
# files.download(checkpoint_dir / "best_model_accuracy=0.6413.pt")


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir "{log_dir}"')


