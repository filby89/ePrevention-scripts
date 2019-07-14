import pandas as pd
import numpy as np
import json
import datetime
import sys
from scipy import stats
import argparse
import seaborn as sns
import os
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from tabulate import tabulate
from collections import OrderedDict
from io import StringIO


class DataType(object):
	"""docstring for DataType"""
	def __init__(self, name, frequency=None, attributes=None, data=None):
		super(DataType, self).__init__()
		self.name = name
		self.frequency = frequency
		self.attributes = attributes
		self.data = data

class ePreventionVisualization(object):
	"""docstring for ePreventionVisualization"""
	def __init__(self, data_path):
		super(ePreventionVisualization, self).__init__()
		self.data_path = data_path


		dtype_dict = {
			"X": np.float64,
			"Y": np.float64,
			"Z": np.float64,
			"accuracy": np.int64,
			"timestamp": np.int64,
			"heartRate": np.int,
			"rRInterval": np.int,
			"totalSteps": np.int,
			"stepsWalking": np.int,
			"stepsRunning": np.int,
			"sleeping": np.int
		}

		linacc = DataType("linacc", 20, ["X", "Y", "Z"])
		gyr = DataType("gyr", 20, ["X", "Y", "Z"])
		hrm = DataType("hrm", 5, ["heartRate", "rRInterval"])
		step = DataType("step", 0, ["totalSteps", "stepsWalking", "stepsRunning"])
		sleep = DataType("sleep", 0, ["sleeping"])
		system = DataType("system")

		self.data_types = OrderedDict({
			"linacc": linacc,
			"gyr": gyr,
			"hrm": hrm,
			"step": step,
			"sleep": sleep,
			"system": system
		})

		LOCALE = "FR"

		for data_type in self.data_types.values():
			f = os.path.join(data_path,data_type.name)
			if os.path.exists(f):
				data_type.data = pd.read_csv(StringIO(open(f).read().replace(",",".")), delimiter=" ", error_bad_lines=False, warn_bad_lines=False)#, dtype=dtype_dict)

		self.fix_timestamps()

		self.get_analytics()



	def get_analytics(self):
		t = []
		for data_type in self.data_types.values():
			if data_type.name != "system" and data_type.name != "step":
				df = data_type.data
				start = df.Timestamp.min()
				end = df.Timestamp.max()

				ideal = (end-start).total_seconds()*data_type.frequency

				true = df.shape[0]

				df['gap'] = (df['Timestamp'].diff()).dt.seconds > 1

				gaps = df[df['gap'] == True].shape[0]

				t.append([data_type.name, start, end, ideal, true, ideal-true, gaps])

		print(tabulate(t, headers=["Type", "Start time", "End time", "Ideal no.", "True no.", "Deviation", "Gaps"]))



	def fix_timestamps(self):
		df_system = self.data_types["system"].data

		LOCALE_LANGUAGE = df_system.loc[df_system.PROPERTY == "LOCALE_LANGUAGE"]

		LAST_BOOT = df_system.loc[df_system.PROPERTY == "START_TIMESTAMP"].iloc[0].VALUE
		UPTIME = df_system.loc[df_system.PROPERTY == "UPTIME"].iloc[0].VALUE

		date = datetime.datetime.strptime(LAST_BOOT, '%Y-%m-%dT%H:%M:%S.%f')
		date = date - datetime.timedelta(seconds=float(UPTIME)) + datetime.timedelta(hours=3)
		for data_type in self.data_types.values():
			if data_type.name != "system" and data_type.name != "step":
				df = data_type.data

				df['Timestamp'] = pd.to_numeric(df['Timestamp'])#.astype(float)*1e-6
				# drop some rows
				# df = df[df['Timestamp']<=df.Timestamp.iloc[-1]]
				# df = df[df['Timestamp']>=df.Timestamp.iloc[0]]

				df['Timestamp'] = pd.to_timedelta(df['Timestamp'], unit="microseconds")
				df['Timestamp'] = df['Timestamp']+date

				# add three hours to all (UTC+3 is greece)
				df['Timestamp'] = df['Timestamp']
				data_type.data = df

			if data_type.name == "step":
				df = data_type.data
				df['start_time'] = pd.to_datetime(df['start_time']) + datetime.timedelta(hours=3)
				df['end_time'] = pd.to_datetime(df['end_time']) + datetime.timedelta(hours=3)

				data_type.data = df
			
			# df.to_csv(os.path.join(self.data_path,data_type.name + "_fixed_times"), index=False, sep=" ", float_format="%.06f")



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, help='Path to folder')
	args = parser.parse_args()

	vis = ePreventionVisualization(data_path=args.path)




