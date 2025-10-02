import torch
import tqdm
import numpy as np
class ModelTrain():
    """
    封装模型训练过程
    data_set 需要实现__call__返回batch_size的数据
    """
    def __init__(self, model, train_loader, validation_loader, test_loader, loss_fn, optimizer, scheduler, recorder, graph):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.recorder = recorder
        self.graph = graph
        self.current_epoch = 0

    def round(self, is_train = True, use_set = 'train', print_summary = False):
        
        self.recorder.clear()
        losses = []

        if use_set == 'train':
            current_loader = self.train_loader
        elif use_set == 'validation':
            current_loader = self.validation_loader
        elif use_set == 'test':
            current_loader = self.test_loader
        else:
            raise ValueError('Wrong set type. use train, validation or test.')

        if is_train:
            self.model.train()
            for batch_x, batch_y in tqdm.tqdm(current_loader):

                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = self.loss_fn(pred, batch_y)
                losses.append(loss.item()) 
                self.recorder.add(pred, batch_y)
                loss.backward()
                self.optimizer.step()
        
        else:
            self.model.eval()
            with torch.no_grad():
                for batch_x, batch_y in tqdm.tqdm(current_loader):
                    pred = self.model(batch_x)
                    loss = self.loss_fn(pred, batch_y)
                    losses.append(loss.item()) 
                    self.recorder.add(pred, batch_y)

        summarys = self.recorder.summary()

        if print_summary:
            return summarys.style.format('{:.2%}')
        else:
            return np.mean(losses), tuple(summarys.iloc[:3,0]), tuple(summarys.iloc[3,1:3])


    def epoch_train(self, epochs, early_stop = 10):

        losses = []

        if self.current_epoch == 0:
            # 如果当前模型刚刚初始化，执行一次测试记录初始损失
            self.graph.reset()
            train_loss, train_summary, train_score = self.round(is_train=False, use_set='train')
            validation_loss, validation_summary, validation_score = self.round(is_train=False, use_set='validation')
            losses.append(validation_loss)
            self.graph.add(self.current_epoch, train_loss, subplot_idx = 0)
            self.graph.add(self.current_epoch, train_summary, subplot_idx = 1)
            self.graph.add(self.current_epoch, train_score, subplot_idx = 2)
            self.graph.add(self.current_epoch, validation_loss, subplot_idx = 3)
            self.graph.add(self.current_epoch, validation_summary, subplot_idx = 4)
            self.graph.add(self.current_epoch, validation_score, subplot_idx = 5)
            self.graph.draw()

        for epoch in range(epochs):

            self.current_epoch += 1
            train_loss, train_summary, train_score = self.round(is_train=True, use_set='train')
            validation_loss, validation_summary, validation_score = self.round(is_train=False, use_set='validation')

            self.scheduler.step()

            losses.append(validation_loss)

            # graph
            self.graph.add(self.current_epoch, train_loss, subplot_idx = 0)
            self.graph.add(self.current_epoch, train_summary, subplot_idx = 1)
            self.graph.add(self.current_epoch, train_score, subplot_idx = 2)
            self.graph.add(self.current_epoch, validation_loss, subplot_idx = 3)
            self.graph.add(self.current_epoch, validation_summary, subplot_idx = 4)
            self.graph.add(self.current_epoch, validation_score, subplot_idx = 5)
            self.graph.draw()
            
            # Early Stop
            if epoch > early_stop:
                if validation_loss>np.mean(losses[-early_stop:]):
                    break

        # 最后输出训练结果表格的对比
        self.round(is_train=False, use_set='train')
        self.recorder.summary()
        self.round(is_train=False, use_set='test')
        self.recorder.summary()

        return self.recorder.summary().iloc[3,0], self.recorder.summary().iloc[3,1] - self.recorder.summary().iloc[3,2]