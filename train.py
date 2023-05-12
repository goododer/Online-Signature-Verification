from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
import torch
import torchvision
import pandas as pd
from config import params
import models
from dataset import SVC2004
import torch.nn.functional as F

#import jason
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class RunBuilder():
    
    @staticmethod # Does not need an instance to use its methods.
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

class RunManager():
    def __init__(self) -> None:
        self.epoch_count = 0
        self.epoch_loss = 0
        # self.epoch_num_correct = 0
        self.epoch_start_time = None # recort the start_time of an epoch.

        self.run_params = None # defined run configrations from RunBuilder.
        self.run_count = 0
        self.run_data = [] # record the results for each epoch.
        self.run_start_time = None

        self.network = None # our model.
        self.loader = None # instance of torch.utils.data.DataLoader
        self.tb = None # Tensorboard.

    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()

        self.run_params = run # namedtuple
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        data = next(iter(self.loader))
        #grid = torchvision.utils.make_grid(data)

        #self.tb.add_image('data', grid)
        self.tb.add_graph(self.network, data)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        print(f'run_{self.run_count} is finished')

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        #self.epoch_num_correct = 0

    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        # accuracy...

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        #self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()

        results['run_ID'] = self.run_count
        results['epoch_ID'] = self.epoch_count
        results['loss'] = loss
        #accuracy
        results['epoch_duration'] = epoch_duration
        results['run_duration'] = run_duration

        for k,v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        #clear_output(wait=True)
        #display(df)

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def save(self, fileName):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{fileName}.csv')
        #with open(f'{fileName}.jason', 'w', encoding = 'utf-8') as f:
        #    jason.dump(self.run_data, f, ensure_ascii=False, indent=4)


if __name__ =='__main__':
    data_dir = './datasets/SVC2004/task1/training/'
    print(f'Running on: {device}')
    m = RunManager()

    for run in RunBuilder.get_runs(params):
        # build make according to configs
        first_encoder = models.First_Encoder(run.window_size, run.channel, run.f_c, run.activation).to(device)
        sequential_encoder = models.Sequential_Encoder(run.f_c, run.activation).to(device)

        first_decoder = models.First_Decoder(run.window_size, run.channel, run.f_c, run.activation).to(device)
        sequential_decoder = models.Sequential_Decoder(run.window_size, run.channel, run.activation).to(device)

        encoder_layer = models.Encoder_layer(first_encoder, sequential_encoder, run.depth).to(device)
        decoder_layer = models.Decoder_layer(first_decoder, sequential_decoder, run.depth).to(device)

        model = models.Encoder_Decoder(encoder_layer, decoder_layer).to(device)

        train_dataset = SVC2004(data_dir, run.max_length, run.window_size)
        loader = DataLoader(train_dataset, run.batch_size)
        optimizer = optim.Adam(model.parameters(), lr = run.lr)

        m.begin_run(run, model, loader)
        for epoch in range(run.epoch):
            m.begin_epoch()
            for batch in loader:
                data = batch
                output = model(data)
                loss = F.mse_loss(data, output)
                optimizer.zero_grad() # zero grad.
                loss.backward()
                optimizer.step()

                m.track_loss(loss)
            m.end_epoch()
            print(f'epoch_{epoch} is finished.')
        m.end_run()

    m.save('results')