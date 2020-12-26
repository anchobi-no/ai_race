
import matplotlib.pyplot as plt
import numpy as np


class TrainingData:
	def __init__(self, n_epoch):
		self.test_result=np.zeros([3,n_epoch])
		self.train_result=np.zeros([3,n_epoch])
		for i in range(n_epoch):
			self.test_result[1:3,i]=None
			self.train_result[1:3,i]=None
		self.init_plot()

	def init_plot(self):
		self.color_1 = "#FF1C05"
		self.color_2 = "#FF059B"
		self.color_3 = "#051EFF"
		self.color_4 = "#0586FF"
		self.fig , self.ax1 = plt.subplots()
		self.ax2 = self.ax1.twinx()
		self.lines = []
		self.lines.append(self.ax1.plot(self.train_result[0,:], self.train_result[1,:], label="train acc", color=self.color_1)[0])
		self.lines.append(self.ax2.plot(self.train_result[0,:], self.train_result[2,:], label="train loss", color=self.color_2)[0])
		self.lines.append(self.ax1.plot(self.test_result[0,:], self.test_result[1,:], label="test  acc", color=self.color_3, linestyle="dashed")[0])
		self.lines.append(self.ax2.plot(self.test_result[0,:], self.test_result[2,:], label="test  loss", color=self.color_4, linestyle="dashed")[0])
		self.ax1.set_ylim(0,1)
		self.ax1.set_xlim(0,101)
		self.ax2.set_ylim(0,0.2)
		self.ax2.set_xlim(0,101)
		self.ax1.set(xlabel="epoch",ylabel="accuracy")
		self.ax2.set(ylabel="loss")
		self.handler1, self.label1 = self.ax1.get_legend_handles_labels()
		self.handler2, self.label2 = self.ax2.get_legend_handles_labels()
		# 凡例をまとめて出力する
		self.ax1.legend(self.handler1 + self.handler2, self.label1 + self.label2,bbox_to_anchor=(1.15,1), loc='upper left', borderaxespad=0.)
		self.fig.subplots_adjust(right=0.7)

	def set_data(self):
		for i in range(2):
			self.lines[i].set_data(self.train_result[0],self.train_result[i+1])
			self.lines[i+2].set_data(self.test_result[0],self.test_result[i+1])
