import torch
import torch.nn as nn


seed = 1
#파이토치 랜덤 시드 고정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Score(nn.Module):
    def __init__(self, state1_dim=5, state2_dim=1, output_dim=1):
        super().__init__()

        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(state1_dim+state2_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)
        self.hidden_act = nn.ReLU()
        self.out_act = nn.Identity()

        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")

    def forward(self, s1, s2):
        x = torch.concat([s1, s2], dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        x = self.out_act(x)
        return x


class Actor(nn.Module):
    def __init__(self, score_net):
        super().__init__()
        self.score_net = score_net

    def forward(self, state1, state2):
        return self.score_net(state1, state2)

    def sampling(self, s1_tensor, portfolio):
        """
        state = (s1_tensor, portfolio)
        s1_tensor: (batch, assets, features)
        """

        for k in range(s1_tensor.shape[1]):
            state2 = portfolio[:,k+1]
            globals()[f"score{k+1}"] = self(s1_tensor[:,k,:], state2)

        for j in range(s1_tensor.shape[1]):
            scores = list() if j == 0 else scores
            scores.append(globals()[f"score{j+1}"])

        batch_num = s1_tensor.shape[0]
        cash_bias = torch.ones(size=(batch_num, 1), device=device) * 0.5
        x = torch.cat(scores, dim=-1)
        x = torch.cat([cash_bias, x], dim=-1)
        x = torch.softmax(x, dim=-1)
        return x


class Critic(nn.Module):
    def __init__(self, score_net, header_dim=None):
        super().__init__()
        self.score_net = score_net
        self.header = Header(input_dim=header_dim)

    def forward(self, state1, state2):
        score = self.score_net(state1, state2)
        v = self.header(score)
        return v


class Header(nn.Module):
    def __init__(self, output_dim=1, input_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128 ,64)
        self.layer3 = nn.Linear(64, output_dim)
        self.hidden_act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, scores):
        x = self.layer1(scores)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        x = self.out_act(x)
        return x

if __name__ == "__main__":
    score_net = Score()
    actor = Actor(score_net)
    critic = Critic(score_net, header_dim=1)

    K = 3
    s1_tensor = torch.rand(size=(10, K, 5))
    portfolio = torch.rand(size=(10, K+1, 1))

    d_portfolio = actor.sampling(s1_tensor, portfolio)
    pi = d_portfolio.view(-1, K+1)[:, 1:]
    v = critic(s1_tensor[:,0], portfolio[:,0])
    print(v)