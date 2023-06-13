import pandas as pd 
import numpy as np
import torch

from Dinger.rebalancing_abstract import Rebalancing
from RL_Rebalancing import Network

class RebalancingAgent(Rebalancing):
    train_date_start = "2014-06-02"
    train_date_end = "2019-12-31"
    test_date_end = "2020-01-02"
    test_date_end = "2021-12-31"
    
    actor_path = "/Users/mac/Desktop/Dinger/Dinger/Models/PPOPortfolio_actor.pth"
    score_path = "/Users/mac/Desktop/Dinger/Dinger/Models/PPOPortfolio_score.pth"

    def __init__(self, num_stock):
        self.num_stock = num_stock
        self.now_portfolio = [0.] * (num_stock + 1)
        self.score_net = Network.Score()
        self.actor = Network.Actor(self.score_net)

        self.actor.load_state_dict(torch.load(self.actor_path, map_location=torch.device('cpu')))
        self.actor.score_net.load_state_dict(torch.load(self.score_path, map_location=torch.device('cpu')))
 
    def desired_portfolio(self, now_data:np.array):
        """
        최적 포트폴리오 산출
        """
        feature_num = now_data.shape[1]

        now_data = torch.FloatTensor(now_data).view(1, self.num_stock, feature_num)
        now_portfolio = torch.FloatTensor(self.now_portfolio).view(1, self.num_stock+1, -1)

        d_portfolio = self.actor.sampling(now_data, now_portfolio).detach()
        d_portfolio = list(d_portfolio[0].numpy())
        d_portfolio = [round(p, 8) for p in d_portfolio]
        self.now_portfolio = d_portfolio
        return d_portfolio

    def desired_action(self, desired_portfolio):
        action = desired_portfolio - self.now_portfolio
        return action

    def rebalancing(self, now_data):
        portfolio = self.desired_portfolio(now_data)
        action = self.desired_action(portfolio)

        print("리밸런싱 포트폴리오:", portfolio, "\n")
        print("필요 거래 비중", action)