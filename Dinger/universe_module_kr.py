import pandas as pd
from datetime import datetime, timedelta

from Dinger.universe_abstract import Universe

'''
df.columns = ['SEC_NM_KOR', 'MKT_VAL', 'ALL_MKT_VAL', 'PER',
'PBR', 'DIV', '매출액', '영업이익', '당기순이익', 'ROA', 'ROE', '베타']
'''

class StockUniverse_KR(Universe):
    
    def __init__(self, data: pd.DataFrame, scaling = None, universe=None, my_portfolio=None):
        self.data = data
        self.data.dropna(inplace=True)
        self.data.종목코드 = self.data.종목코드.astype(str).str.zfill(6).tolist()
        self.universe = universe
        self.my_portfolio = my_portfolio
        self.scaling = scaling

    def get_standard_df(self, mode='mean'):
        """
        mode = 'mean', 'logistic'
        database module output df
        mode  : 어떤  기준 ex. mean, median으로 할지
        """

        if '종목코드' in self.data.columns:
            self.data.set_index('종목코드', inplace=True)
    
        if mode == 'mean':
            return self.data.groupby('SEC_NM_KOR').mean()

        if mode == 'quantile':
            return self.data.groupby('SEC_NM_KOR').quantile()

        if mode == 'median':
            return self.data.groupby('SEC_NM_KOR').median()

    def get_score(self, mode='mean', scaling=None):
        """
        mode = 'mean','quant','median'
        """
        if '종목코드' in self.data.columns:
            self.data.set_index('종목코드', inplace=True)

        ### Scaling
        tmp = self.data.drop(['CMP_KOR', 'SEC_NM_KOR'], axis=1)

        if scaling == 'mm':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(tmp)
            scaled_data = pd.DataFrame(scaled_data, columns=tmp.columns).set_index(tmp.index)

            scaled_data[['PER', 'PBR', '베타']] = scaled_data[['PER', 'PBR', '베타']] * (-1)
            score = scaled_data.sum(axis=1).to_dict()

            for i in self.data.index:
                self.data.loc[i, 'Score'] = score[i]

            return self.data

        if scaling == 'std':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(tmp)
            scaled_data = pd.DataFrame(scaled_data, columns=tmp.columns).set_index(tmp.index)
            scaled_data[['PER', 'PBR', '베타']] = scaled_data[['PER', 'PBR', '베타']] * (-1)
            score = scaled_data.sum(axis=1).to_dict()

            for i in self.data.index:
                self.data.loc[i, 'Score'] = score[i]

            return self.data

        sector_list = self.data.SEC_NM_KOR.unique().tolist()
        if (mode == 'mean') or (mode == 'quant') or (mode == 'median'):
            if mode == 'mean':
                sec_standard = self.data.groupby('SEC_NM_KOR').mean()
            if mode == 'quant':
                sec_standard = self.data.groupby('SEC_NM_KOR').quantile()
            if mode == 'median':
                sec_standard = self.data.groupby('SEC_NM_KOR').median()


            score = dict()
            for sec in sector_list:
                tmp = self.data.loc[(self.data.SEC_NM_KOR == sec)].drop(['CMP_KOR', 'SEC_NM_KOR'], axis=1)
                for i in tmp.index.tolist():
                    score[i] = 0
                    for col in tmp.columns:
                        if col in ['PER', 'PBR', '베타']:
                            score[i] += (tmp.loc[i][col] < sec_standard.loc[sec, col])
                        else:
                            score[i] += (tmp.loc[i][col] > sec_standard.loc[sec, col])

            for i in self.data.index:
                self.data.loc[i, 'Score'] = score[i]
            return self.data



    def top_n_sectors(self, n=5, mode='mean', scaling=None):
        """
        데이터 베이스에서 가져온 종목들의 점수를 계산
        """
        score = self.get_score(mode=mode).groupby('SEC_NM_KOR').mean().sort_values('Score', ascending=False)
        return score.iloc[:n]


    def get_universe(self, n=5, mode='mean', sector=None, scaling=None, bonus=False):
        """
        input sector 에 따라 top n stocks 가져오기
        예를 들어 it만 입력할 수 있고 , 3개 넣을 수도 있음. list로 입력
        투자 유니버스내 기업 총 개수: 120개 - 국가 (2) x 산업 (11) x 상위 기업(5)
        """
        if sector is None:
            sector = self.data.SEC_NM_KOR.unique()
        
        result = pd.DataFrame()
        for sec in sector:
            score = self.get_score(mode=mode)
            score = score.loc[score.SEC_NM_KOR == sec].sort_values('Score', ascending=False)[:n]
            result = pd.concat([result, score])

        if bonus:
            self.universe = self.get_resnet()
        else:
            self.universe = result
        return self.universe

    def get_initial_portfolio(self, n=5, mode='mean',sector=None):
        """
        input : universe
        n개 sector 의 n개 stocks
        3개 sector 1~2개 stock
        """
        my_universe = self.get_universe(n=n, mode=mode, sector=sector)
        self.my_portfolio = my_universe
        return my_universe

    def get_portfoilo_ohlcv(self, initial_portfolio, ratio=False):
        '''
        가장 최근 ohlcv 하나만 가져오기
        :return:
        '''
        from datetime import datetime, timedelta
        from pykrx import stock
        import numpy as np

        tickers = initial_portfolio.index.astype(str).str.zfill(6).tolist()
        start = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        end = datetime.now().strftime('%Y%m%d')
        initial_ohlcv_dict = {}

        for ticker in tickers:
            if ratio:
                ohlcv = stock.get_market_ohlcv(start, end, ticker, 'd').pct_change()
            else:
                ohlcv = stock.get_market_ohlcv(start, end, ticker, 'd')
            initial_ohlcv_dict[ticker] = np.array(ohlcv.iloc[-1])

        return initial_ohlcv_dict


    def get_resnet(self, start=(datetime.now() - timedelta(days=120)).strftime('%Y%m%d'),
                   end=datetime.now().strftime('%Y%m%d'), dmy='d'):
        import numpy as np
        import pandas as pd
        import torch
        from pykrx import stock
        from Dinger import resnet

        if self.universe.empty == True:
            print('포트폴리오를 구성해주세요')
            pass

        resnet_dict = {}
        universe = self.universe
        error = []
        universe.index = universe.index.astype(str).str.zfill(6)

        for ticker in universe.index.tolist():
            try:
                data = stock.get_market_ohlcv(start, end, ticker, dmy)

                # 가장 최근 64 길이 (이미지 1장)
                test_data = data.iloc[-64:][["종가", "거래량"]].dropna()

                # model
                model = resnet.ResNet()

                # model weight load
                model.load_state_dict(torch.load("./Dinger/Models/resnet.pth"))

                # 추론 이미지 1장 (0: Up, 1: Down, 2: Side)
                # Input : pd.DataFrame
                # Output: torch.Tensor
                pred = model.predict(test_data)
                pred = np.array(pred)[0] * (-1) + 1
                if pred < 0:
                    pred = 0.5
                resnet_dict[ticker] = pred
            except:
                error.append(ticker)
                pass

        for k in resnet_dict:
            universe.loc[k, 'resnet'] = resnet_dict[k]

        universe['Score'] += universe.resnet
        self.universe = universe
        return universe







