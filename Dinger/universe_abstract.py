from abc import ABCMeta, abstractmethod
class Universe(metaclass=ABCMeta):


    @abstractmethod
    def get_standard_df(self):
        """
        database module output df 
        mode  : 어떤  기준 ex. mean, median으로 할지 
        """
        pass

    @abstractmethod
    def get_score(self):
        """
        데이터 베이스에서 가져온 종목들의 점수를 계산
        data = {fdsfsf}
        """
        pass

    @abstractmethod
    def top_n_sectors(self):
        """
        데이터 베이스에서 가져온 종목들의 점수를 계산
        data = {fdsfsf}
        """
        pass

    @abstractmethod
    def get_universe(self):
    
        """
        input sector 에 따라top n stocks 가져오기 
        예를 들어 it만 입력할 수 있고 , 3개 넣을 수도 있음. list로 입력
        투자 유니버스내 기업 총 개수: 120개 - 국가 (2) x 산업 (11) x 상위 기업(5)

        """ 
        pass

    @abstractmethod
    def get_initial_portfolio(self):
    
        """
        input : universer 
        n개 sector 의 n개 stocks 
        3개 sector 1~2개 stock
        """
        pass