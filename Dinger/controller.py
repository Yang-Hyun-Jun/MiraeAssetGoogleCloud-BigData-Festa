from os.path import isfile
from Dinger.universe_module_us import StockUniverse_US
from Dinger.universe_module_kr import StockUniverse_KR
from Dinger.database_module_us import stockUS
from Dinger.database_module_kr import stockKR
from Dinger.rebalancing_module import RebalancingAgent
from Dinger import info
from IPython.display import display
from datetime import datetime

import os
import shutil
import numpy as np
import pandas as pd 
import FinanceDataReader as fdr

class Controller:
    """
    Dinger Controller
    
    mode : "mean" or "quant" or "median"
    """

    MAIN_STATEMENT = "명령어를 입력하세요. (h: 도움말): "

    def __init__(self, mode="mean"):
        
        # 데이터 저장 폴더 생성
        if not os.path.isdir(info.paths["fundamental_path"]):
            os.makedirs(info.paths["fundamental_path"], exist_ok=True)
        
        if not os.path.isdir(info.paths["score_path"]):
            os.makedirs(info.paths["score_path"], exist_ok=True)

        if not os.path.isdir(info.paths["standard_path"]):
            os.makedirs(info.paths["standard_path"], exist_ok=True)
        
        if not os.path.isdir(info.paths["top_sector_path"]):
            os.makedirs(info.paths["top_sector_path"], exist_ok=True)

        if not os.path.isdir(info.paths["universe_path"]):
            os.makedirs(info.paths["universe_path"], exist_ok=True)

        if not os.path.isdir(info.paths["portfolio_path"]):
            os.makedirs(info.paths["portfolio_path"], exist_ok=True)

        # 펀더멘탈 데이터 수집 (저장 파일 없을 경우)
        if not os.path.isfile(info.paths["fundamental_path"] + '/fundamental_kr.csv'):
            print("한국 펀더메달 데이터 수집 필요. 수집을 시작합니다.")
            kr_database = stockKR('20221005')
            funda_kr, _ = kr_database.stocks_info()
            funda_kr.to_csv(info.paths["fundamental_path"] + '/fundamental_kr.csv')
        
        if not os.path.isfile(info.paths["fundamental_path"] + '/fundamental_us.csv'):
            print("미국 펀더메달 데이터 수집 필요. 수집을 시작합니다.")
            sp500_list = fdr.StockListing('S&P500')
            us_database = stockUS(sp500_list)
            us_database.jsonMaker()
            funda_us, _ = us_database.stocks_info()
            funda_us.to_csv(info.paths["fundamental_path"] + '/fundamental_us.csv')

        self.mode = mode
        self.terminating = False
        self.command_list = []
        self.empty = True
        self.create_command()

        self.stock_us = pd.read_csv(info.paths["fundamental_path"] + '/fundamental_kr.csv', index_col=0)
        self.stock_kr = pd.read_csv(info.paths["fundamental_path"] + '/fundamental_us.csv')

        self.uni_US = StockUniverse_US(self.stock_us)
        self.uni_KR = StockUniverse_KR(self.stock_kr)

        self.standard_US = self.uni_US.get_standard_df(mode=self.mode)
        self.standard_KR = self.uni_KR.get_standard_df(mode=self.mode)


    def create_command(self):
        """ 명령어 정보를 생성한다. """
        self.command_list = [
            {
                "guide": "{0:15} 도움말 출력".format("h, help"),
                "cmd": ["help"],
                "short": ["h"],
                "action": self.print_help,
            },
            {
                "guide": "{0:15} 스코어 데이터 프레임 조회 및 저장".format("s, score"),
                "cmd": ["score"],
                "short": ["s"],
                "action": self.get_score_df,
            },
            {
                "guide": "{0:15} 스코어링 기준값 데이터 프레임 조회 및 저장".format("d, standard"),
                "cmd": ["standard"],
                "short": ["d"],
                "action": self.get_standard_df,
            },
            {
                "guide": "{0:15} 상위 섹터 데이터 프레임 조회 및 저장".format("n, nsector"),
                "cmd": ["nsector"],
                "short": ["n"],
                "action": self.get_top_n_sectors_df,
            },
            {
                "guide": "{0:15} 유니버스 데이터 프레임 조회 및 저장".format("u, universe"),
                "cmd": ["universe"],
                "short": ["u"],
                "action": self.get_universe_df,
            },
           {
                "guide": "{0:15} 최초 포트폴리오 데이터 프레임 조회 및 저장".format("p, portfolio"),
                "cmd": ["portfolio"],
                "short": ["p"],
                "action": self.get_initial_portfolio_df,
            },
           {
                "guide": "{0:15} 최초 포트폴리오 리밸런싱 후 기록 저장".format("r, rebalancing"),
                "cmd": ["rebalancing"],
                "short": ["r"],
                "action": self.get_rebalanced,
            },
            {
                "guide": "{0:15} 프로그램 종료".format("t, terminate"),
                "cmd": ["terminate"],
                "short": ["t"],
                "action": self.terminate,
            },
            {
                "guide": "{0:15} 최초 포트폴리오 초기화".format("i, initialize"),
                "cmd": ["initialize"],
                "short": ["i"],
                "action": self.init_portfolio,
            },
        ]
    

    # def get_profitloss(self):


    def main(self):
        while not self.terminating:
            
            key = input(self.MAIN_STATEMENT)
            self._on_command(key)

    def print_help(self):
        """ 가이드 문구 출력 """
        print("명령어 목록 ==============")
        for item in self.command_list:
            print(item["guide"])
    
    def init_portfolio(self):
        """ 보유 포트폴리오 초기화 """
        shutil.rmtree(info.paths["portfolio_path"])
        self.empty = True
        info.now_portfolio = None

    def get_standard_df(self):
        """ 스코어링 기준값 데이터 저장 및 조회 """
        country = input("한국 시장은 0 입력, 미국 시장은 1 입력 :")
        verbose = input("데이터 저장은 0 입력, 데이터 출력은 1 입력 :")
        
        if country == "0":
            if verbose == "0":
                print("한국 시장의 기준값 데이터 프레임을 저장합니다.")
                self.standard_KR.to_csv(info.paths["standard_path"] + "/standard_KR.csv")
            elif verbose == "1":
                print("한국 시장의 기준값 데이터 프레임을 조회합니다.")
                display(self.standard_KR)

        if country == "1":
            if verbose == "0":
                print("미국 시장의 기준값 데이터 프레임을 저장합니다.")
                self.standard_US.to_csv(info.paths["standard_path"] + "/standard_US.csv")
            elif verbose == "1":
                print("미국 시장의 기준값 데이터 프레임을 조회합니다.")
                display(self.standard_US)


    def get_score_df(self):
        """ 스코어 데이터 저장 및 조회 """
        country = input("한국 시장은 0 입력, 미국 시장은 1 입력 :")
        verbose = input("데이터 저장은 0 입력, 데이터 출력은 1 입력 :")
        
        if country == "0":
            score_data = self.uni_KR.get_score()
            if verbose == "0":
                print("한국 시장의 스코어 데이터 프레임을 저장합니다.")
                score_data.to_csv(info.paths["score_path"] + "/score_KR.csv")
            elif verbose == "1":
                print("한국 시장의 스코어 데이터 프레임을 조회합니다.")
                display(score_data)
        
        if country == "1":
            score_data = self.uni_US.get_score()
            if verbose == "0":
                print("미국 시장의 스코어 데이터 프레임을 저장합니다.")
                score_data.to_csv(info.paths["score_path"] + "/score_US.csv")
            elif verbose == "1":
                print("미국 시장의 스코어 데이터 프레임을 조회합니다.")
                display(score_data)    


    def get_top_n_sectors_df(self):
        """ 상위 n개 섹터 저장 및 조회 """
        country = input("한국 시장은 0 입력, 미국 시장은 1 입력 :")
        n = int(input("섹터 개수 입력: "))
        verbose = input("데이터 저장은 0 입력, 데이터 출력은 1 입력 :")
        
        if country == "0":
            top_n_sectors = self.uni_KR.top_n_sectors(n)
            if verbose == "0":
                print("한국 시장의 상위 섹터 데이터 프레임을 저장합니다.")
                top_n_sectors.to_csv(info.paths["top_sector_path"] + f"/top_{str(n)}_sectors_kr.csv")
            elif verbose == "1":
                print("한국 시장의 상위 섹터 데이터 프레임을 조회합니다.")
                display(top_n_sectors)
        
        if country == "1":
            top_n_sectors = self.uni_US.top_n_sectors(n)
            if verbose == "0":
                print("미국 시장의 상위 섹터 데이터 프레임을 저장합니다.")
                top_n_sectors.to_csv(info.paths["top_sector_path"] + f"/top_{str(n)}_sectors_us.csv")
            elif verbose == "1":
                print("미국 시장의 상위 섹터 데이터 프레임을 조회합니다.")
                display(top_n_sectors)    


    def get_universe_df(self):
        """ 유니버스 저장 및 조회 """
        country = input("한국 시장은 0 입력, 미국 시장은 1 입력 :")
        n = int(input("종목 개수 입력 : "))
        verbose = input("데이터 저장은 0 입력, 데이터 출력은 1 입력 :")
        
        sector_list_kr = list(self.stock_kr["SEC_NM_KOR"].unique())
        sector_list_us = list(self.stock_us["Sector"].unique())

        # 한국 시장
        if country == "0":
            sectors = []
            sector = None
            while sector != "0":
                sector = input("한국 유니버스 구성 섹터를 입력하시오: (완료는 0, 섹터 조회는 1)")
                if sector == "1":
                    print("===입력 가능 섹터 리스트===")
                    print(sector_list_kr)
                if sector not in sector_list_kr and sector not in ["0", "1"]:
                    print("존재하지 않는 섹터입니다.")
                elif sector in sector_list_kr and sector not in ["0", "1"]:
                    sectors.append(sector)

            if not sectors:
                print("입력 섹터가 없습니다.")
                return 
            else:    
                universe = self.uni_KR.get_universe(sector=sectors, n=n)
                if verbose == "0":
                    print("한국 시장의 유니버스 데이터 프레임을 저장합니다.")
                    universe.to_csv(info.paths["universe_path"] + "/universe_kr.csv")
                elif verbose == "1":
                    print("한국 시장의 유니버스 데이터 프레임을 조회합니다.")
                    display(universe)    

        # 미국 시장
        if country == "1":
            sectors = []
            sector = None
            while sector != "0":
                sector = input("미국 유니버스 구성 섹터를 입력하시오: (완료는 0, 섹터 조회는 1)")
                if sector == "1":
                    print("===입력 가능 섹터 리스트===")
                    print(sector_list_us)
                if sector not in sector_list_us and sector not in ["0", "1"]:
                    print("존재하지 않는 섹터입니다.")
                elif sector in sector_list_us and sector not in ["0", "1"]:
                    sectors.append(sector)

            if not sectors:
                print("입력 섹터가 없습니다.")
                return 
            else:    
                universe = self.uni_US.get_universe(sectors=sectors, num=n)
                if verbose == "0":
                    print("미국 시장의 유니버스 데이터 프레임을 저장합니다.")
                    universe.to_csv(info.paths["universe_path"] + "/universe_us.csv")
                elif verbose == "1":
                    print("미국 시장의 유니버스 데이터 프레임을 조회합니다.")
                    display(universe)    


    def get_initial_portfolio_df(self):
        """ 최초 포트폴리오 저장 및 조회 """
        verbose = input("데이터 저장은 0 입력, 데이터 출력은 1 입력 :")

        if verbose == 0 and not self.empty:
            print("보유 포트폴리오가 이미 존재합니다.")
            return

        sectors_kr = []
        sectors_us = []

        sector_list_kr = list(self.stock_kr["SEC_NM_KOR"].unique())
        sector_list_us = list(self.stock_us["Sector"].unique())

        sector = None
        while sector != "0":
            sector = input("한국 포트폴리오 구성 섹터를 입력하시오: (완료는 0, 섹터 조회는 1)")
            if sector == "1":
                print("===입력 가능 섹터 리스트===")
                print(sector_list_kr)
            if sector not in sector_list_kr and sector not in ["0", "1"]:
                print("존재하지 않는 섹터입니다.")
            if sector in sector_list_kr and sector not in ["0", "1"]:
                sectors_kr.append(sector)

        sector = None
        while sector != "0":
            sector = input("미국 포트폴리오 구성 섹터를 입력하시오: (완료는 0, 섹터 조회는 1)")
            if sector == "1":
                print("===입력 가능 섹터 리스트===")
                print(sector_list_us)
            if sector not in sector_list_us and sector not in ["0", "1"]:
                print("존재하지 않는 섹터입니다.")
            if sector in sector_list_us and sector not in ["0", "1"]:
                sectors_us.append(sector)

        if sectors_kr:
            n_kr = int(input("한국 포트폴리오 종목 개수를 입력하시오 :"))
            port_stocks_kr = self.uni_KR.get_initial_portfolio(sector=sectors_kr, n=n_kr)
            if verbose == "0":
                print("한국 시장의 최초 포트폴리오 데이터 프레임을 저장합니다.")
                port_stocks_kr.to_csv(info.paths["portfolio_path"] + "/port_stocks_kr.csv")
                self.empty = False
            elif verbose == "1":
                print("한국 시장의 최초 포트폴리오 데이터 프레임을 조회합니다.")
                display(port_stocks_kr)

        if sectors_us:
            n_us = int(input("미국 포트폴리오 종목 개수를 입력하시오."))
            port_stocks_us = self.uni_US.get_initial_portfolio(sectors=sectors_us, num_stocks=n_us)
            if verbose == "0":
                print("미국 시장의 최초 포트폴리오 데이터 프레임을 저장합니다.")
                port_stocks_us.to_csv(info.paths["portfolio_path"] + "/port_stocks_us.csv")
                self.empty = False
            elif verbose == "1":
                print("미국 시장의 최초 포트폴리오 데이터 프레임을 조회합니다.")
                display(port_stocks_us)

                        
    def get_portfolio_ohlcv(self, ratio=False):
        """ 최초 포트폴리오의 가격 정보 조회 """
        
        kr_price_info = {}
        us_price_info = {}

        if os.path.isfile(info.paths["portfolio_path"] + "/port_stocks_kr.csv"):
            port_stocks_kr = pd.read_csv(info.paths["portfolio_path"] + "/port_stocks_kr.csv", index_col=0)
            kr_price_info = self.uni_KR.get_portfoilo_ohlcv(port_stocks_kr, ratio=ratio)

        if os.path.isfile(info.paths["portfolio_path"] + "/port_stocks_us.csv"):
            port_stocks_us = pd.read_csv(info.paths["portfolio_path"] + "/port_stocks_us.csv", index_col=0)
            us_price_info = self.uni_US.get_portfolio_ohlcv(port_stocks_us, ratio=ratio)

        pd.options.display.float_format = '{:.5f}'.format
        index = ["Open", "High", "Low", "Close", "Volume"]
        all_price_info = dict(kr_price_info, **us_price_info)
        all_price_info = pd.DataFrame(all_price_info, index=index)

        if not dict(all_price_info):
            print("최초 포트폴리오 데이터가 없습니다.")
            return
        else:
            return all_price_info
        

    def get_rebalanced(self):
        """ 리밸런싱 후 기록 """
        num_features = 5
        all_price_info = self.get_portfolio_ohlcv(ratio=True)
        name_assets = ["Cash"] + list(all_price_info.columns)
        all_price_info = all_price_info.to_numpy().reshape(-1, num_features)
        num_stocks = all_price_info.shape[0]
        
        agent = RebalancingAgent(num_stocks)

        if info.now_portfolio != None:
            agent.now_portfolio = info.now_portfolio 

        rebalanced_portfolio = agent.desired_portfolio(now_data=all_price_info)
        info.now_portfolio = rebalanced_portfolio

        # 저장 
        pd.options.display.float_format = '{:.5f}'.format
        today = datetime.today().strftime('%Y-%m-%d-%H-%M')
        portfolio_info = pd.DataFrame(columns=name_assets, index=[today])
        portfolio_info.iloc[0] = rebalanced_portfolio

        if os.path.isfile(info.paths["portfolio_path"] + "/portfolio_note.csv"):
            portfolio_note = pd.read_csv(info.paths["portfolio_path"] + "/portfolio_note.csv", index_col=0)
            portfolio_note = pd.concat([portfolio_note, portfolio_info])
            portfolio_note.to_csv(info.paths["portfolio_path"] + "/portfolio_note.csv")
        else:
            portfolio_info.to_csv(info.paths["portfolio_path"] + "/portfolio_note.csv")

        return rebalanced_portfolio


    def terminate(self):
        """ 프로그램 종료 """
        print("프로그램 종료 중 ....")
        self.terminating = True
        print("프로그램 종료")


    def _on_command(self, key):
        """ 커맨드 처리를 담당 """
        for cmd in self.command_list:
            if key.lower() in cmd["cmd"] or key.lower() in cmd["short"]:
                print(f"{cmd['cmd'][0].upper()} 명령어를 실행합니다.")
                cmd["action"]()
                return
        print("잘못된 명령어가 입력 되었습니다.") 

