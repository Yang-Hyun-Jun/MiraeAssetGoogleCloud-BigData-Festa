#Abstract class for KR/US stock database 
#last updated on 09/20/2022

from abc import *
from abc import ABCMeta, abstractmethod

class StockDatabase(metaclass=ABCMeta): 


    @abstractmethod
    def crawling(self):

        """
        website로 부터 stock 종목의 quantatitive 정보를 가져오는 크롤링 method 
        """
        pass


    @abstractmethod
    def get_fundamentals(self):

        """
        
        크롤링해서 가져온 데이터에서 분석에 사용할 fundamentals 추출 
        
        -PER(주가수익비율) = 주가 / EPS = 시가총액/주식수 / 순이익/주식수 = 시가총액 / 순이익 
        -PBR(주당순자산비율) = 주가 / BPS = 시가총액/주식수 / 순자산/주식수 = 시가총액 / 순자산 
        -ROE(자기자본수익률): 당기순이익 / 자기자본 * 100
        -Beta 지수 :
        주식 수익의 안정성을 확인해 보기 위한 지표로써 sharpe 지수를 사용
        -순이익, 영업이익, 매출액, 시가총액

        """
        pass 

    @abstractmethod
    def stocks_info(self):

        """
        list에 기제된 stock 종목들 fundamentals를 pandas dataframe 생성 
        """
        pass 
