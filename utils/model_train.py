import torch
import tqdm
import numpy as np
class ModelTrain():
    """
    封装模型训练过程
    data_set 需要实现__call__返回batch_size的数据
    """
    def __init__(self, model, train_set, validation_set, test_set, loss_fn, optimizer, scheduler, recorder, graph, threshold):
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.recorder = recorder
        self.graph = graph
        self.current_epoch = 0
        self.threshold = threshold

    def round(self, round, is_train = True, use_set = 'train', print_summary = False):
        
        self.recorder.clear()
        losses = []

        if use_set == 'train':
            current_set = self.train_set
        elif use_set == 'validation':
            current_set = self.validation_set
        elif use_set == 'test':
            current_set = self.test_set
        else:
            raise ValueError('Wrong set type. use train, validation or test.')

        if is_train:
            self.model.train()
            for i in tqdm.tqdm(range(round)):

                batch_data = current_set(batch_size = 10)
                batch_x, batch_y = batch_data[:-1], batch_data[-1]

                self.optimizer.zero_grad()
                pred = self.model(*batch_x)
                loss = self.loss_fn(pred, batch_y)
                losses.append(loss.item()) 
                self.recorder.add(pred[:,1:], batch_y[:,:1])
                loss.backward()
                self.optimizer.step()
        
        else:
            self.model.eval()
            with torch.no_grad():
                for i in tqdm.tqdm(range(round)):

                    batch_data = current_set(batch_size = 10)
                    batch_x, batch_y = batch_data[:-1], batch_data[-1]

                    pred = self.model(*batch_x)
                    loss = self.loss_fn(pred, batch_y)
                    losses.append(loss.item()) 
                    self.recorder.add(pred[:,1:], batch_y[:,:1])

        if print_summary:
            return self.recorder.summary(threshold = self.threshold)

        return np.mean(losses), self.recorder.distribution(), self.recorder.average_score()


    def epoch_train(self, epochs, round, early_stop = 10):

        losses = []

        if self.current_epoch == 0:
            # 如果当前模型刚刚初始化，执行一次测试记录初始损失
            self.graph.reset()
            train_loss, train_summary, train_score = self.round(round = round, is_train=False, use_set='train')
            validation_loss, validation_summary, validation_score = self.round(round = round, is_train=False, use_set='validation')
            losses.append(validation_loss)
            self.graph.add(self.current_epoch, train_loss, subplot_idx = 0)
            self.graph.add(self.current_epoch, train_summary, subplot_idx = 1)
            self.graph.add(self.current_epoch, train_score, subplot_idx = 2)
            self.graph.add(self.current_epoch, validation_loss, subplot_idx = 3)
            self.graph.add(self.current_epoch, validation_summary, subplot_idx = 4)
            self.graph.add(self.current_epoch, validation_score, subplot_idx = 5)

        for epoch in range(epochs):

            self.current_epoch += 1
            train_loss, train_summary, train_score = self.round(round = round, is_train=True, use_set='train')
            validation_loss, validation_summary, validation_score = self.round(round = round, is_train=False, use_set='validation')

            self.scheduler.step()

            losses.append(validation_loss)

            # graph
            self.graph.add(self.current_epoch, train_loss, subplot_idx = 0)
            self.graph.add(self.current_epoch, train_summary, subplot_idx = 1)
            self.graph.add(self.current_epoch, train_score, subplot_idx = 2)
            self.graph.add(self.current_epoch, validation_loss, subplot_idx = 3)
            self.graph.add(self.current_epoch, validation_summary, subplot_idx = 4)
            self.graph.add(self.current_epoch, validation_score, subplot_idx = 5)

            # Early Stop
            if epoch > early_stop:
                if validation_loss>np.mean(losses[-early_stop:]):
                    break

        # 最后输出训练结果表格的对比
        self.round(round = round, is_train=False, use_set='train')
        self.recorder.summary()
        self.round(round = round, is_train=False, use_set='test')
        self.recorder.summary()