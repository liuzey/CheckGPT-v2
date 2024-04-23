import sys, os, time

# log recorder

class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        # output_dir = "./75_logpool/"  # folder
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        #log_name = '{}.txt'.format(time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
        # log_name_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        # log_name = log_name_time + ".txt"
        self.filename = filename  # os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(self.filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass