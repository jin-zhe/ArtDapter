import datetime

class Stopwatch():
	TIME_UNITS = ['hour', 'minute', 'second']


	def __init__(self):
		self.reset()


	def reset(self):
		self.start = datetime.datetime.now()


	def __call__(self):
		elapsed_seconds = int((datetime.datetime.now() - self.start).total_seconds())
		minutes, seconds = divmod(elapsed_seconds, 60)
		hours, minutes = divmod(minutes, 60)
		return hours, minutes, seconds


	def __repr__(self):
		return Stopwatch.format_time(*self())


	@staticmethod
	def format_time(hours, minutes, seconds):
		def get_str(x, unit):
			if x != 0:
				return f"{x} {unit + ('s' if x > 1 else '')}"
			else:
				return ''
		seq = [get_str(x,unit) for x,unit in zip([hours, minutes, seconds], Stopwatch.TIME_UNITS)]
		seq = [x for x in seq if x]  # remove empty strings
		return ' '.join(seq)
