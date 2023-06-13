from torch import nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(Model, self).__init__()
        # 1 hidden layer and softmax output
        # loss function: cross entropy
        self.layers = nn.Sequential(
            # nn.Dropout(0.5),
            # nn.Linear(input_dim, 62),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.5),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LogSoftmax(dim=1),
        )
        # choose device based on availability
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # NVIDIA GPU
            print('ANN Using CUDA GPU: ', torch.cuda.get_device_name(0))
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device('mps') # Apple GPU
        #     print('ANN Using Apple GPU')
        else:
            self.device = torch.device('cpu') # CPU
            print('ANN Using CPU')
        self.to(self.device)
        
    def forward(self, x):
        logits = self.layers(x)
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores
    
    def predict(self, x):
        return self.forward(x).argmax(dim=1)
    
    def loss(self, x, y):
        return nn.NLLLoss()(self.forward(x), y)
    
    def accuracy(self, x, y):
        return (self.predict(x) == y).float().mean()
    
    def train(self, x, y, optimizer, x_test, y_test, epochs=100, batch_size=32):
        print('Training on', self.device)
        print('X shape:', x.shape)
        print('Y shape:', y.shape)
        # convert Y to one-hot
        # y = F.one_hot(y, num_classes=10)
        # y_test = F.one_hot(y_test, num_classes=10)
        # print('Y shape:', y.shape)
        # print(y[0])
        # print('Y test shape:', y_test.shape)
        x = x.to(self.device)
        y = y.to(self.device)
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        losses = []
        accuracies = []
        test_accuracies = []
        for epoch in range(epochs):
            for i in range(0, x.shape[0], batch_size):
                # forward
                loss = self.loss(x[i:i+batch_size], y[i:i+batch_size])
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch}: loss {loss.item()}')
            losses.append(loss.item())
            accuracies.append(self.accuracy(x, y).item())
            test_accuracies.append(self.accuracy(x_test, y_test).item())
        print(f'Train Accuracy: {self.accuracy(x, y)}')
        # print(f'Test Accuracy: {self.accuracy(x_test, y_test)}')
        return losses, accuracies, test_accuracies
        
    def test(self, x, y):
        print(f'Test Accuracy: {self.accuracy(x.to(self.device), y.to(self.device))}')
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        torch.load_state_dict(torch.load(path))