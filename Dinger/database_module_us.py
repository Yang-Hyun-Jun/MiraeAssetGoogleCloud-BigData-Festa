
##===database module for US stock 

#last updated on 10/11/2022
##==update historyㅌ
#10/11/2022 : conv to float 함수명 수정 
#10/03/2022 : add ohlcv maker
#09/23/2020 : add *super()
#09/20/2022 : initialize


from Dinger.utils import *
from Dinger.database_abstract import *

from datetime import datetime, timedelta
import yfinance as yf


# us stock data base module 

class stockUS(StockDatabase):

    def __init__(self, stock_list : pd.DataFrame ):

        """
        initialize the class with stock list
        input : 분석하고자 하는 종목들이 dataframe 
        (df에 column명 'Symbol' = ticker , 'Name' = 종목명 이어야 함)
        utils file의 fdr API 사용해서 fdr.StockListing('')로 호출 가능 
        """
        self.stock_list = stock_list


    
    def crawling(self, sym):

        """
        입력한 ticker의 정보를 finbiz website에서 크롤링해 출력 

        input : ticker 
        output : finbiz website의 해당 종목 quantative 정보 

        """
        super().crawling()

        #=== crawling from finbiz website 

        headers = {'User-Agent': 'Mozilla/5.0'} 
        # w.o header: 403 - Forbidden: Access is denied

        print("ticker:",sym)
        
        r = requests.get('https://finviz.com/quote.ashx?t='+ sym, headers=headers)
        soup = BeautifulSoup(r.text, features="lxml")
        
        tables = soup.find_all('table') # bring all table tag 
        snapshot_table2 = soup.find('table', attrs={'class': 'snapshot-table2'})
        tables = pd.read_html(str(snapshot_table2))
        df = pd.DataFrame(tables[0])

        # convert df to 1 column per each ticker 
        df.columns = ['key', 'value'] * 6
        df_list = [df.iloc[:, i*2 : i*2+2] for i in range(6)] #convert columns to rows 
        df_factor = pd.concat(df_list, ignore_index = True)
        df_factor.set_index('key', inplace= True)

        return df_factor


    def __convToFloat(self, s):
        
        """
        crawling 한 값들의 measurement 통일 
        """

        if s[-1] == '%':
            s = s.replace('%', '') 
        if s[-1] in list('BMK'): # covert dollar measurement
            powers = {'B': 10 ** 9, 'M': 10 ** 6, 'K': 10 ** 3, '': 1}
            m = re.search("([0-9\.]+)(M|B|K|)", s)
            """
            re.serach(condition, string)- find the condition in the string 
            condition 
            (pattern) - find pattern and save mathced data 
            [filter]+ = find 0-9 , expreess any character behind as escape , + ( find more than one character satisfying the filter condition)
            (|||) one of M, B, K can come 
            """
            if m:
                val, mag = m.group(1), m.group(2) #gorup(1) -  ([0-9\.]+), group(2) -(M|B|K|)
                return float(val) * powers[mag] 
        try:
            result = float(s)

        except:
            result = None
        
        return result


    
    def get_fundamentals(self, sym):

        """

        크롤링해서 가져온 데이터에서 분석에 사용할 fundamentals 추출 

        """
        super().get_fundamentals()
        v = self.crawling(sym).value

        sales = v['Sales']
        revenue = v['Income']
        marcap = v['Market Cap']
        per = v['P/E']
        pbr = v['P/B']
        roa = v['ROA']
        roe = v['ROE']
        volume =v['Volume']
        price = v['Price']
        beta = v['Beta']
        dividend = v['Dividend %']
        

        sales = self.__convToFloat(sales)
        revenue = self.__convToFloat(revenue)
        marcap = self.__convToFloat(marcap)
        per = self.__convToFloat(per)
        pbr = self.__convToFloat(pbr)
        roa = self.__convToFloat(roa)
        roe = self.__convToFloat(roe)
        volume = self.__convToFloat(volume)
        price = self.__convToFloat(price)
        beta  = self.__convToFloat(beta)
        dividend = self.__convToFloat(dividend)

        fundamentals = sales, revenue, marcap, per, pbr, roa, roe, volume, price, beta, dividend

        return fundamentals
 



    def _assignSector(self):

        """
        stock list에  sector가 없는 경우 (eg. industry만 있는 경우), SP500 을 기준으로 Assign
        저장된 sector - industry mapping의 industry명은 fdr nyse list기준으로 다른 리스트를 사용할 시 추가 점검 필요 
        """

        stock_list = self.stock_list
        stock_list_tmp = stock_list
        error_list = []

        if 'Sector' in stock_list.columns :
            print("Sector 정보가 이미 존재합니다")
            stock_list.head()
            pass 

        else:

            standard_sector = {'Health Care': ['헬스케어 업체 및 서비스','제약', '헬스케어 장비 및 용품', '생명과학 및 메디컬 리서치', ],
                    'Information Technology': ['반도체 및 반도체 장비','소프트웨어 및 IT서비스','전자 장비 및 부품','통합 하드웨어 및 소프트웨어'],
                    'Energy' : ['오일 및 가스','오일 및 가스 장비 및 서비스','재생에너지' ],
                    'Consumer Staples' :['식품 및 약품소매', '개인, 가정 용품 및 서비스', '개인, 가정 용품 및 서비스' ,'음료', '식품 및 담배','다양한 소매업', '사무기기', '가정용품'],
                    'Industrials' : ['복합 기업', '화물 및 물류 서비스','우주항공 및 국방', '전문 및 상업 서비스', '건설장비 및 대형차, 기차, 배','주택건설','운송업','건설 및 엔지니어링', '운송 인프라','다양한 산업재 도매업'],
                    'Financials' : ['은행', '투자은행 및 서비스', '보험', '핀테크', '다양한 금융서비스','기타 금융업' ],
                    'Consumer Discretionary' : ['개인, 가정 용품 및 서비스', '자동차 및 자동차 부품','호텔 및 엔터테인먼트','섬유 및 의류','컴퓨터, 전화 및 가전제품','레저 상품','전문품 상점'],
                    'Communication Services' : ['미디어 및 출판' , '통신 서비스','커뮤니케이션 및 네트워킹'],
                    'Utilities' : ['전기 유틸리티', '복합 유틸리티','수도 유틸리티','천연가스 유틸리티'],
                    'Materials' : ['화학','메탈&마이닝','건축 자재','용기 및 포장','우라늄','종이 및 임산물','석탄'], 
                    'Real Estate' : ['주택 및 상업용 리츠','부동산 운영 관리'],
                    'ETC' :['교육서비스','학교','정부활동','비즈니스교육']
                    }


            for content in list(stock_list_tmp.itertuples()):
                try: 
                    tmp_sec = [k for k,v in standard_sector.items() if content.Industry in v]
                    stock_list_tmp.loc[content.Index, "Sector"] =  tmp_sec

                except: 
                    print("ERROR FOR", content)
                    tmp_sec = None
                    error_list.append(content)
            
            self._assignSector =  stock_list_tmp


        return stock_list_tmp , error_list


    def jsonMaker(self, folder = 'sp500'):

        """
        추출한 fundamentals 정보를 종목별로 json파일로 저장  
        저장 directory : 현재 work directory 내 객체 생성시 입력한 stock_list명으로 폴더 생성
        
        return 값을 확인해서 오류 확인 가능 
        return : val_error_list, imp_error_list 
        주로 stock_list 내 ticker와 finbiz website의 ticker 명이 일치하지 않을 경우 val_error가 발생함.  

        """
        DIR = './'
        folder = f'{folder}_factors/'
        #folder = "sp500_factors/"

        if not os.path.isdir(DIR+ folder):
            os.mkdir(DIR + folder)

        stock_list = self.stock_list

        val_error_list = []
        imp_error_list = []

        for ix, row in stock_list.iterrows(): 
            sym, name = row['Symbol'] , row['Name'] #ticker, company name 
            json_fn = folder + f'{sym}.json'
            if os.path.exists(json_fn):
                print('skip', json_fn)
                continue
            
            else:
                try: 
                    factors = self.get_fundamentals(sym)

                except ValueError:
                    val_error_list.append(sym)
                    continue  

                except ImportError:
                    imp_error_list.append(sym)
                    continue

                with open(json_fn, 'w') as f:
                    json.dump(factors, f)
                    print(sym, name)
                    
        return val_error_list, imp_error_list
 

    
    def stocks_info(self, folder = 'sp500'):

        """

        종목별 fundamentals가 담긴 json files들을 모아 데이터 frame 생성 
        input = folder , json file이 담겨있는 folder명 입력 
        ouptut = dataframe 
        """
        super().stocks_info()

        DIR = './'
        folder = f'{folder}_factors/'
        #folder = "sp500_factors/"


        tmp_list = self.stock_list.copy()
        error_list = []
        for ix, row in tmp_list.iterrows():
            sym, name = row['Symbol'], row['Name']
            json_fn = DIR + folder + f'{sym}.json'


        # open json files saved in json maker function
        # if there's an error in opening json file, save as none
            try: 
                with open(json_fn, 'r') as f:
                    factors = json.load(f)

        #make another df with fundamental data
                for col in ["Sales","Revenue","MarCap", "PER", "PBR", "ROA", "ROE", "Volume", "Price", "Beta", "DIVIDEND"]:
                    tmp_list.loc[ix,col] = factors[col]
            except:
                for col in ["Sales","Revenue","MarCap", "PER", "PBR", "ROA", "ROE", "Volume", "Price", "Beta", "DIVIDEND"]:
                    tmp_list.loc[ix,col] = None
                error_list.append(sym)    
                print("error for", sym, name)
        
        df = tmp_list #fundamental 정보가 담긴 df 
                
                
        return df, error_list

    
    def ohlcv_maker(stocklist, start_date = None, folder = None):

        """
        입력한 stock list의 ohlcv data를 가져오는 function입니다.

        """

        DIR = './'

        if folder == None : 
            folder = "ohlcv/"

        if not os.path.isdir(DIR+ folder):
            os.mkdir(DIR + folder)


        if start_date == None : 
            start_date = datetime.now() - timedelta(days=128) # 64 *2 일전
            
        tmp_list = stocklist.copy()
        for _ , row in tmp_list.iterrows():
            sym= row['Symbol']
            
            #이미 저장된 symbol은 skip
            if os.path.exists(DIR + folder + f'{sym}_ohlcv.csv'):
                pass
            
            #yf에서 주가 정보 가져오기 

            tmp_olhcv = yf.download(sym, start_date )

            #csv로 저장 
            tmp_olhcv.to_csv(DIR + folder + f'{sym}_ohlcv.csv')

        print("종목별 최신 ohlcv를 csv로 저장했습니다.")
        

