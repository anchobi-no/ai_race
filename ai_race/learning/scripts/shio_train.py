'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
from matplotlib import cm
import os
import io
import datetime
import threading
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ShioDataSet import MyDataset
from samplenet_analog import *
from trainingdata import *
from testreport import *

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../config")
import learning_config

DISCRETIZATION = learning_config.Discretization_number

Trepo = TestReport()
def main():
	# Parse arguments.
	args = parse_args()

	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	ROOT_DIR = ""
	
	imgDataset = MyDataset(args.data_csv, ROOT_DIR, 320, 240, transform=transforms.ToTensor(), PILtrans = args.PILtrans, ORGtrans = args.ORGtrans)
	# Load dataset.
	train_data, test_data = train_test_split(imgDataset, test_size=0.2)
	#pd.to_pickle(test_data, "test_data.pkl")
	#del test_data
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
	
	print('data set')
	# Set a model.
	if args.model == 'resnet18':
		model = models.resnet18()
		model.fc = torch.nn.Linear(512, DISCRETIZATION)
	elif args.model == 'samplenet':
		model = SampleNet()
	elif args.model == 'simplenet':
		model = SimpleNet()
	else:
		raise NotImplementedError()
	model.train()
	model = model.to(device)

	print('model set')
	# Set loss function and optimization function.
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
	#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	print('optimizer set')
	
	Tdata = TrainingData(args.n_epoch)

	# Train and test.
	print('Train starts')
	for epoch in range(args.n_epoch):
		# Train and test a model.
		train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
		Tdata.train_result[0,epoch]=epoch+1
		Tdata.train_result[1,epoch]=train_acc
		Tdata.train_result[2,epoch]=train_loss

		# Output score.
		if(epoch%args.test_interval == 0):
			#pd.to_pickle(train_data, "train_data.pkl")
			#del train_data
			
			#test_data = pd.read_pickle("test_data.pkl")
			test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)
			#del test_data
			test_acc, test_loss = test(model, device, test_loader, criterion, epoch+1)
			Tdata.test_result[0,int(epoch/args.test_interval)]=epoch+1
			Tdata.test_result[1,int(epoch/args.test_interval)]=test_acc
			Tdata.test_result[2,int(epoch/args.test_interval)]=test_loss
			#del test_loader
			
			stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
			print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))
			global Trepo
			for repo in Trepo.report:
				if repo["epoch"]==epoch+1:
					print("")
					#print(json.dumps(repo, indent=4))
			#train_data = pd.read_pickle("train_data.pkl")
		else:	
			stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}' #, test acc: {:<8}, test loss: {:<8}'
			print(stdout_temp.format(epoch+1, train_acc, train_loss)) #, test_acc, test_loss))
		Tdata.set_data()
		plt.pause(1)
		# Save a model checkpoint.
		if(epoch%args.save_model_interval == 0):
			model_ckpt_path = (savedir+"checkpoints/{}_{}_epoch={}.pth").format(args.dataset_name, args.model_name, epoch+1)
			torch.save(model.state_dict(), model_ckpt_path)
			print('Saved a model checkpoint at {}'.format(model_ckpt_path))
			print('')
	
	global settings
	with open(savedir+'settings.json','w') as f:
		json.dump(settings, f, indent=4)

	with open(savedir+'reports.json','w') as f:
		json.dump(Trepo.report, f, indent=4)

	np.savetxt(savedir+'testdata.csv', Tdata.test_result.transpose(), delimiter=',', fmt='%f')
	np.savetxt(savedir+'traindata.csv', Tdata.train_result.transpose(), delimiter=',', fmt='%f')
	Tdata.fig.savefig(savedir+'traindata.png')



def train(model, device, train_loader, criterion, optimizer):
	model.train()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		# Forward processing.
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, targets)

		# Backward processing.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Set data to calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()

		# Calculate score at present.
		train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)
		if batch_idx % 10 == 0 and batch_idx != 0:
			stdout_temp = 'batch: {:>3}/{:<3}, train acc: {:<8}, train loss: {:<8}'
			print(stdout_temp.format(batch_idx, len(train_loader), train_acc, train_loss))

	# Calculate score.
	train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)

	return train_acc, train_loss


def test(model, device, test_loader, criterion, epoch_n):
	model.eval()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(test_loader):
		# Forward processing.
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		
		# Set data to calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()
		
	test_acc, test_loss = calc_score(output_list, target_list, running_loss, test_loader)

	print('confusion_matrix')
	conf_mat = confusion_matrix(output_list, target_list)
	print(conf_mat)
	print('classification_report')
	class_repo = classification_report(output_list, target_list)
	print(class_repo)

	global Trepo
	Trepo.append(epoch_n,class_repo,conf_mat.tolist())

	return test_acc, test_loss



def calc_score(output_list, target_list, running_loss, data_loader):
	# Calculate accuracy.
	#result = classification_report(output_list, target_list) #, output_dict=True)
	#acc = round(result['weighted avg']['f1-score'], 6)
	acc = round(f1_score(output_list, target_list, average='micro'), 6)
	loss = round(running_loss / len(data_loader.dataset), 6)

	return acc, loss


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument("--dataset_name", type=str, default='sim_race')
	arg_parser.add_argument("--data_csv", type=str, default=os.environ['HOME'] + '/Images_from_rosbag/_2020-11-05-01-45-29_2/_2020-11-05-01-45-29.csv')
	arg_parser.add_argument("--model", type=str, default='resnet18')
	arg_parser.add_argument("--model_name", type=str, default='joycon_ResNet18')
	arg_parser.add_argument("--model_ckpt_dir", type=str, default=os.environ['HOME'] + '/work/experiments/models/')
	arg_parser.add_argument("--model_ckpt_path_temp", type=str, default=os.environ['HOME'] + '/work/experiments/models/checkpoints/{}_{}_epoch={}.pth')
	arg_parser.add_argument('--n_epoch', default=20, type=int, help='The number of epoch')
	arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
	arg_parser.add_argument('--test_interval', default=5, type=int, help='test interval')
	arg_parser.add_argument('--save_model_interval', default=5, type=int, help='save model interval')
	arg_parser.add_argument('--PILtrans', action='store_true')
	arg_parser.add_argument('--ORGtrans', action='store_true')

	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.model_ckpt_dir, exist_ok=True)
	dat = datetime.datetime.now()
	cur_time = str(dat.year).zfill(4)+str(dat.month).zfill(2)+str(dat.day).zfill(2)+str(dat.hour).zfill(2)+str(dat.minute).zfill(2)+str(dat.second).zfill(2)
	
	global savedir
	savedir = args.model_ckpt_dir+args.model_name+"_"+cur_time+"/"

	os.makedirs(savedir, exist_ok=True)
	os.makedirs(savedir + "checkpoints/", exist_ok=True)

	global settings
	settings={"dataset_name"	:args.dataset_name
			 ,"data_csv"		:args.data_csv
			 ,"model"			:args.model
			 ,"model_name"		:args.model_name
			 ,"model_ckpt_dir"	:args.model_ckpt_dir
			 ,"model_ckpt_path_temp":args.model_ckpt_path_temp
			 ,"n_epoch"			:args.n_epoch
			 ,"lr"				:args.lr
			 ,"test_interval"	:args.test_interval
			 ,"save_model_interval"	:args.save_model_interval
			 ,"PILtrans"		:args.PILtrans
			 ,"ORGtrans"		:args.ORGtrans
			 }

	print(args.data_csv)
	# Validate paths.
	assert os.path.exists(args.data_csv)
	assert os.path.exists(args.model_ckpt_dir)

	return args


if __name__ == "__main__":
	main()
