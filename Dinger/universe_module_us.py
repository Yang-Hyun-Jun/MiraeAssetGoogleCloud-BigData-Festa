
##==== initial portfolio 만들기 
#last updated on 10/14/2022
##==update history 
#10/11/2022 : 순서 변경, 
#10/11/2022 : add resnet score after getting universe
#10/09/2022 : change ohlcv output form sipmple values to to ohlcv ratio form previous date 
#10/04/2022 : added scaling option and get_portfolio_ohlcv
#10/03/2022 : added cnn score and methods accordingly
#09/30/2022 : complete method (~ get_initial_portfolio)
#09/29/2022 : add abstract method
#09/23/2022 : initialize


"""
추가 수정사항 
scored df- sacling 적용시 negative list 빼게 
get universe에서 cnn nan값으로 나오는 것 오류확인 후 수정 

"""

"""
Table
    - Standard DF생성 
    - 종목 scoring 
    ( 상위섹터, 상위종목 검색 가능)
    -Universe 구성 , CNN score 추가 (option값으로 확인 가능 )
    - Initial Portfolio 구성 

"""

import yfinance as yf
from sklearn.preprocessing  import MinMaxScaler, StandardScaler
from Dinger.utils import *
from Dinger.universe_abstract import *
from Dinger.resnet import * #cnn score를 구할 package import 


class StockUniverse_US(Universe): 

    def __init__(self, df = pd.DataFrame, scaling = None):

        """
        initialize class 

        Input
        df : stocks들의 fundamental 정보가 담긴 df (database module에서 생성, 다른 위치에 저장된 것을 불러옴)
        scaling: 점수화 때 적용할 scaling (mm: minmax, std :standard scaling )

        default values 
        attribute list for scoring 
        positive attirbutes : 값이 높을 수록 좋은 features 
        negative attribtues : 값이 낮을 수록 좋은 fuatures

        """

        self.df = df
        self.p_attribute_list = ['Sales','Revenue','MarCap','ROA','ROE']
        self.n_attribute_list = ['PER']
        self.scaling = scaling 
        self.universe = None




    def get_standard_df(self, mode = 'mean', q = 0.5 , df = None) : 

        """

        stock 종목을 점수화할 기준이 될 standard df 생성 (Sector별)

        Input
        mode: (mean, median, quantile) , standard df 를 생성할 기준 
        q: mode - quantile 선택 시 quantile 값 

        Output 
        dataframe : 점수화 기준이 되는 df

        """

        super().get_standard_df()

        if df == None : 
            df = self.df # class initilaize시 입력했던 stock fundamental 정보가 담긴 dataframe

        if mode == 'mean': 
            self.standard_df = pd.DataFrame(df.groupby(['Sector']).mean())

        elif mode == 'median' :
            self.standard_df = pd.DataFrame(df.groupby(['Sector']).median())

        elif mode == 'quantile' :
            self.standard_df = pd.DataFrame(df.groupby(['Sector']).quantile(q))

        return self.standard_df
    



    def get_score(self, df = None, standard_df = None, p_attribute_list = None, n_attribute_list=None, scaling = None) :

    
        """

        Standard df 를 기준으로 각 항목에 점수 매기는 method 
        scaling option값에 따라 구하는 방식이 달라짐.
        1. (default) :Sector 별 mean 값과 비교해 높으면 1, 낮으면 0
        2. 'mm' : feature scaling 후 단순 합 
        3. 'std' : feature scaling 후 단순 합 

        Input 
        df : default 값 (class initialize한 fundamental 정보 담긴 df)
        standard_df : Sector별 standard df (default, get_standard_df에서 구한 df)
        scaling : scoring 시 feature를 scaling 할 지 여부 
        p_attribute_list : 높을수록 좋은 features list 
        n_attribute_list : 낮을수록 좋은 features list 
        defalut 값 
            self.p_check_list = ['Sales','Revenue','MarCap','ROA','ROE']
            self.n_check_list = ['PER']

        output: dataframe (종목별 fundamental  정보 + "att"_score)

        """

        super().get_score()

        if df == None : 
            df = self.df
        if standard_df == None :
            standard_df = self.standard_df


        if p_attribute_list == None:
            p_attribute_list = self.p_attribute_list
        if n_attribute_list == None: 
            n_attribute_list = self.n_attribute_list
        
        scored_df = df.copy()

        if scaling == None : # scaling 없을 시 

            for idx , row in scored_df.iterrows(): # standdard df 와 비교 후 크면 1, 작으면 0으로 score 매기기 
                for att in p_attribute_list:
                    if row[att] >= standard_df.loc[row['Sector'], att] :
                        scored_df.loc[idx, att + '_score'] = 1
                    else: 
                        scored_df.loc[idx, att + '_score'] = 0
                    
                for att in n_attribute_list : 
                    if row[att] <= standard_df.loc[row['Sector'], att] :
                        scored_df.loc[idx, att + '_score'] = 1
                    else: 
                        scored_df.loc[idx, att + '_score'] = 0


            scored_df['total_score'] = scored_df[scored_df.columns[scored_df.columns.str.contains('_score')]].sum(axis = 1)
        
            
            return scored_df

        import numpy as np 

        if scaling == "mm": #minmax scaler
            data = scored_df.iloc[:,4:] # ticker, industry등 string value 제외
            cols = list(scored_df.columns[4:])

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            scaled_data = pd.DataFrame(scaled_data, columns = cols).set_index(scored_df.index)

            """
            n_attribute_list 차 구하는 데에서 계속 오류나서 일단 합으로만 함.

            시도 series간 차. sub, -
            -abs 합 등등

            """

            tmp= scaled_data[p_attribute_list].sum(axis=1) 
            #tmp = scaled_data[n_attribute_list]

            #print("=====tmp", type(tmp), tmp)
            #sub_tmp = tmp- pd.Series(scaled_data[n_attribute_list])
            #print("subtmp",sub_tmp)

            #tmp = scaled_data[p_attribute_list].sum(axis=1).sub(scaled_data[n_attribute_list])
            #print(tmp)
            scaled_data.loc[:,'total_score'] = tmp
        
            
            mm_scaled_scored_df = scored_df.iloc[:,:4].merge(scaled_data, right_index= True, left_index = True) # merge fundamental data + scaled score

            scored_df = mm_scaled_scored_df

            return scored_df
                
            
        if scaling == "std": #standardscaler
            data = scored_df.iloc[:,4:] # ticker, industry등 string value 제외
            cols = list(scored_df.columns[4:])
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            scaled_data = pd.DataFrame(scaled_data, columns = cols).set_index(scored_df.index)
            # 시도 했으나 실패 
            tmp = scaled_data[p_attribute_list].sum(axis=1).sub(scaled_data[n_attribute_list])
            scaled_data.loc[:,'total_score'] = tmp
            #scaled_data.assign( total_score = tmp)

            #tmp = scaled_data[p_attribute_list].sum(axis=1).sub(scaled_data[n_attribute_list])
            #scaled_data.assign( total_score = tmp)
            
            std_scaled_scored_df = scored_df.iloc[:,:4].merge(scaled_data, right_index= True, left_index = True)

            scored_df = std_scaled_scored_df
            
            return scored_df




    def top_n_sectors(self, num, scaling = None) :

        """
        total score 기준으로 가장 점수가 높은 num개의 sector추출 
        입력한 scaling에 따라 method를 호출 해 score를 생성 

        Input 
        num : 상위 몇개의 sector를 추출할지 
        scaling : 점수합의 scaling 방법 지정. (default: None)
        
        Output
        df

        """

        super().top_n_sectors()

        scored_df = self.get_score(scaling = scaling)

        tmp = pd.DataFrame(scored_df.groupby(['Sector']).sum().sort_values(by = "total_score", ascending = False))

        top_sectors = tmp[:num]

        return top_sectors


  
    def __top_n_stocks(self, sectors, num, scored_df = None, scaling = None):

        """

        private method

        total score 기준으로 입력한 sectors에서 가장 점수가 높은 num개의 stocks 추출 
        scored_df 를 따로 입력하지 않을 시 scaling에 따라 get_score method를 호출 해 scored df를 생성 

        Input 
        secotrs : stock을 추출할 sectors
        num : 상위 몇개의 stocks를 추출할지 
        scaling : 점수합의 scaling 방법 지정. (default: None)
        
        Output
        Dataframe
        입력한 sector의 top n 개 종목 dataframe 
        
        """

        if scored_df is None :
            scored_df = self.get_score(scaling = scaling)
    
        top_stocks = []

        for sector in range(len(sectors)):
            tmp = scored_df[scored_df['Sector'] == sectors[sector]].sort_values(by = 'total_score', ascending = False)
            sym = tmp.Symbol[:num].to_list()
            top_stocks = top_stocks + sym


        df = scored_df[scored_df['Symbol'].isin(top_stocks)].sort_values(by='Sector')
    
        
        return df #각 sector별 top n stocks들어있는 dictionary




    def get_cnn_score( self, data: pd.DataFrame , size = 64, epochs = 10):

        """
        only for one stock 

        resnet을 이용해 한 개 stock의 64일치 OHLCV를 통해 추후 4일의 up or down trend 예측

        Input
        data : stock 의 ohlcv data [] , pd.DataFrame에 "Close", "Volume" column이 있어야 함 
        size : 몇 일치 데이터를 보고 예측 할 건지. default 64
        epochs : default 10

        Output
        dictonary {ticker: predcition}

        """

        cnn_score = dict()

        data = data[["Close", "Volume"]] #clsoe와 volume만 가져오기 

        gd = GenerateDataset(data = data, size = size)
        image_data, label_data = gd.generate_image()
        convnet = ResNet()
        training_model = TrainingModel(model=convnet, x_all=image_data, y_all=label_data)
        training_model.train_test_split()
        training_model.train(epochs=epochs)
        training_model.test()

        i = 0
        if np.argmax(label_data[i]) == 0:
            pred = 1 ,
            title ="Uptrend" # uptrend
        elif np.argmax(label_data[i]) == 1:
            pred = 0 ,
            title ="Downtrend"# uptreand
        else:
            pred = 0.5 ,
            title ="Sidetrend" #sidetrend

        cnn_score["추세예측값"] = pred

        #plot
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title(title)
        plt.imshow(image_data[i,0], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(image_data[i+4,0], cmap="gray")
        plt.show()

        self.cnn_score = cnn_score

        #모델 저장 
        folder = './content'
        path = f"./content/ResNet-{epochs}epochs.pth"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        training_model.save(path)

        return self.cnn_score #dictionary 


    def _get_many_cnn_score(self, path = 'sp500', size = 64, epochs = 10 , df = None):
        
        """
        resnet을 이용해 여러개 stock의 64일치 OHLCV를 통해 추후 4일의 up or down trend 예측

        Input
        path : 종목의 ohlcv가 저장 돼 있는 folder명 (string) eg.sp500_ohlcv이면 sp500입력 - database module에서 생성
        size : 몇 일치 데이터를 보고 예측 할 건지. default 64
        epochs : default 10
        df: cnn score를 구하고자 하는 list가 담기 df (default = universe df )

        Output
        cnn dictionary { symbol : prediction}

        """

        DIR = f'./{path}_ohlcv/'

        cnn_score = dict()

        for sym in df.Symbol :
            #symbol 에 따른 data가져오기 

            try: 
                data = pd.read_csv(DIR+ f'{sym}_ohlcv.csv', index_col = 0) #각 symbol의 데이터 가져오기 
                data = data[["Close", "Volume"]] #clsoe와 volume만 가져오기

                gd = GenerateDataset(data = data, size = size, info = False)
                image_data, label_data = gd.generate_image(info = False)
                convnet = ResNet()
                training_model = TrainingModel(model=convnet, x_all=image_data, y_all=label_data)
                training_model.train_test_split()
                training_model.train(epochs=epochs)
                training_model.test()

                i = 0
                if np.argmax(label_data[i]) == 0:
                    pred = 1 # uptrend
                elif np.argmax(label_data[i]) == 1:
                    pred = 0 # downtreand
                else:
                    pred = 0.5 #sidetrend

                #모델 저장 
                folder = './content'
                path = f"./content/ResNet-{epochs}epochs.pth"
                if not os.path.isdir(folder):
                    os.mkdir(folder)

                training_model.save(path)

            except: 
                pred = None

            cnn_score[sym] = pred

            
        #convert cnn score dictionary to cnn_score_df 
        self.cnn_score = cnn_score #dictionary
        
        
        # 따로 df -> csv 로 저장할 시 사용 
        # cnn_score_df = pd.DataFrame.from_dict(cnn_score, orient= 'index')

        return self.cnn_score


    
    def __add_cnn_score(self, path= 'sp500' , scored_df = None , dict = None ):

        """
        cnn score dictionary를 get universe df에 추가한다 

        Input

        path : ohlcv data가 들어있는 folder name (database 모듈에서 생성) 
        scored_df: 종목들이 점수화 완료 된 df (scaling mode 에 따라 변수 따로 지정해서 입력 가능, 
        입력 안할 시 가장 마지막에 산출한 scaling mode에 따른 df 호출)
        dict: cnn score dictionary

        Output
        Dataframe : cnn score가 추가된 df
        """

        #추가 확인 필요 (dict= 직접 인자로 안넣었을 때 작동안됨ㄴ)

        if scored_df is None: 
            scored_df = self.scored_df

        if dict is None:
            dict = self._get_many_cnn_score(path = path , df = scored_df) # 추출하고 싶은 항목
        


    #sym 을 보고 추가 _ 추후에는  get_score method에 모두 추가할 수 있을 듯 
        for k, v in dict.items():
            for idx, row in scored_df.iterrows():
                if row.Symbol == k:
                    scored_df.loc[idx, 'cnn_score'] = v

        self.scored_df = scored_df
    
        return self.scored_df





    def get_universe(self, sectors , num, scored_df = None, scaling = None, bonus =False):

        """
        total score 기준으로 입력한 sectors에서 가장 점수가 높은 num개의 stocks 추출해서 universe 생성
        eg.투자 유니버스내 기업 총 개수: 120개 - 국가 (2) x 산업 (11) x 상위 기업(5) 
        scored_df 를 따로 입력하지 않을 시 입력한 scaling에 따라 get_score method를 호출 해 scored df를 생성 

        Input
        sectors : 어떤 섹터에서 universe를 추출할 지 리스트로 입력 eg. [Information Technology], ['Industrials', 'Information Technology']
        num : 정한 섹터 중, top 몇개 종목을 가져올 지 입력합니다.

        Output
        df : 입력한 sector의 각 sector에서 top n개 종목 추출한 dataframe 
        """

        super().get_universe()


        universe = self.__top_n_stocks(sectors = sectors , num = num, scaling = scaling, scored_df= scored_df) 

        if bonus : #resnet 추가 add cnn 점수 보여줌
            universe = self.__add_cnn_score(scored_df = universe)

        self.universe= universe

        return self.universe


    def get_initial_portfolio(self, sectors , num_stocks, path= 'sp500' , dict = None , scaling = None):

        """
        input : 
        universe dataframe , sector list , n개 stocks 
        path = ohlcv data가 들어있는 folder name  
    
        output : dataframe dictionary {Sector : [[sym, stock name],[sym, stock name]], Sector2 : [[sym, stock name],[sym, stock name]]}
        """


        super().get_initial_portfolio()
        if self.universe is not None:
            universe = self.universe
        else:
            universe = self.get_universe(sectors=sectors, num=num_stocks)
        

        if len(sectors) > universe.Sector.nunique() : 
            raise ValueError("portfolio secotors의 수는 universe sectors의 수보다 클 수 없습니다.")
        
        #get universer에서 cnn 이미 추가 안했었을 시 
        if 'cnn_score' in universe.columns :
            pass
        else:
            universe = self.__add_cnn_score(scored_df = universe, path= 'sp500')

        
        #추가한 cnn score 반영해서 total score 업데이트 

        universe.loc[:,'new_total_score'] = universe[['cnn_score','total_score']].sum(axis=1) #update total score including cnn_score
 
        #new_total_score에 따라 universe추출
        
        top_stocks = []

        for sector in range(len(sectors)):
            tmp = universe[universe['Sector'] == sectors[sector]].sort_values(by = 'new_total_score', ascending = False)
            sym = tmp.Symbol[:num_stocks].to_list()
            top_stocks = top_stocks + sym

        
        display_cols = ['Symbol', 'Name', 'Sector', 'Industry', 'Sales', 'Revenue', 'MarCap',
       'PER', 'PBR', 'ROA', 'ROE', 'Volume', 'Price', 'Beta', 'DIVIDEND','new_total_score']


        initial_port = universe[universe['Symbol'].isin(top_stocks)].sort_values(by = 'Sector')
        initial_port = initial_port[display_cols]
        self.initial_port = initial_port
        return self.initial_port

    def get_portfolio_ohlcv(self, initial_portfolio, ratio=False):

        """
        입력한 포트폴리오의 ohlcv 데이터 변화율 dictionary 로 반환 
        output: dictionary ((v_t - v_(t-1)) / v_(t-1) )
        example
        { 종목: [ d, d, d, d, d], 종목2: [ d, d, d, d, d], 종목3:[ d, d, d, d, d] }

        """

        port_dict = {}
        sectors = list(initial_portfolio["Sector"].unique())

        for idx in range(len(sectors)):

            tmp = initial_portfolio[initial_portfolio.Sector == str(sectors[idx])]
            tmp_sym = tmp.Symbol.tolist()
            tmp_name = tmp.Name.tolist()
            port_dict[sectors[idx]] = [pair for pair in zip(tmp_sym, tmp_name)]

        # initial portfolior 각 sector의 (val) value list 내에서 vals[i] 첫번째 [0] == sym 가져오기   
        tmp = [vals[i][0] for vals in port_dict.values() for i in range(len(vals))] 
        
        initial_ohlcv_dict = {}
        
        # yf 에서 ohlcv가져오기 

        for ticker in range(len(tmp)):
            tmp_ohlcv = yf.download(tmp[ticker], period = '5d') # show recent 5 output
            # print("=====",ticker, "최신 ohlcv date:",tmp_ohlcv.index[0],"=====") #해당 ticker의 최신 ohlcv 추출 날짜 
            recent = tmp_ohlcv.loc[:,['Open','High','Low','Close','Volume']].values[-1] # yfinance에 있는 해당 종목 가장 최근 OHLCV값 추출
            recent_next = tmp_ohlcv.loc[:,['Open','High','Low','Close','Volume']].values[-2] # 2nd 최근값
            ohlcv_ratio = (recent - recent_next) / recent_next # ( v_t - v_(t-1) ) / v_(t-1)

            if ratio:
                initial_ohlcv_dict[tmp[ticker]] = ohlcv_ratio
            else:
                initial_ohlcv_dict[tmp[ticker]] = recent

        return initial_ohlcv_dict


        

