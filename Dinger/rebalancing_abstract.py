from abc import ABCMeta, abstractmethod

class Rebalancing(metaclass=ABCMeta):
    
    def desired_portfolio(self):
        """
        RL actor network로부터
        최적 Rebalanced portfolio를 output
        """