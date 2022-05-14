# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import numpy as np
import pandas as pd
import datetime,time
import logging

from sklearn import svm

def get_logger():
    logger = logging.getLogger("industry_svc")
    logger.setLevel(logging.INFO)

    today = time.strftime('%Y%m%d', time.localtime(time.time()))
    fileHandler = logging.FileHandler("d://cauchy/logs/industry_svc/isvc_"+ today +".log", mode='a')
    fileHandler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s - line:%(lineno)d - [%(levelname)s] : %(message)s")
    fileHandler.setFormatter(formatter)

    consoleHandler = logging.StreamHandler()
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    return logger
logger = get_logger()
'''
行业轮动 + 机器学习（支持向量机）结合策略
逻辑：1.在行业指数标的中，选取历史收益最大一个行业指数，选取其中的N个绩优股
'''
# 策略中必须有init方法
def init(context):
    # 待轮动的行业指数(分别为：300工业.300材料.300可选.300消费.300医药.300金融)
    # 300医药：SHSE.000913 农业：SHSE.000122 300地产：SHSE.000952 基建：SHSE.000950
    context.industry_index = ['SHSE.000913', 'SHSE.000122', 'SHSE.000952', 'SHSE.000950']
    # 用于统计数据的天数
    context.stat_days = 5


    # 历史窗口长度
    context.svm_history_len = 10
    # 预测窗口长度
    context.svm_forecast_len = 4
    # 训练样本长度
    context.svm_training_len = 90 #20天为一个交易月

    # 持股数量
    context.holding_num = 3
    # 止盈幅度
    context.earn_rate = 0.06
    # 最小涨幅卖出幅度
    context.sell_rate = 0.02
    # 止损幅度
    context.loss_rate = 0.05
    # 最长持股天数
    context.holding_days = 12

    # 预测任务
    schedule(schedule_func=chooseIndustryStock, date_rule='1d', time_rule='14:45:00')
    # 订阅行情任务，用于执行建仓策略
    schedule(schedule_func=subFunc, date_rule='1d', time_rule='09:31:00')



# 订阅行情方法
def subFunc(context):
    logger.info("subFuc start")
    positions = context.account().positions()
    symbols = ",".join([position['symbol'] for position in positions])
    logger.info("订阅行情，当前持有:{}".format(symbols))
    subscribe(symbols=symbols, frequency='60s', unsubscribe_previous=True)


    # 根据行业指数获取绩优股
def chooseIndustryStock(context):
    logger.info("chooseIndustryStock start")
    now = context.now

    positions = context.account().positions()
    if len(positions) >= context.holding_num:
        logger.info("仓位已满，直接退出.")
        return

    # 获取上一个交易日
    last_day = get_previous_trading_date(exchange='SHSE', date=now)
    return_index = pd.DataFrame(columns=['return'])
    # 获取并计算行业指数收益率
    for i in context.industry_index:
        return_index_his = history_n(symbol=i, frequency='1d', count=context.stat_days + 1, fields='close,bob',
                                     fill_missing='Last', adjust=ADJUST_PREV, end_time=last_day, df=True)
        return_index_his = return_index_his['close'].values
        return_index.loc[i, 'return'] = return_index_his[-1] / return_index_his[0] - 1

    # 获取指定数内收益率表现最好的行业
    sector = return_index.index[np.argmax(return_index)]
    logger.info('{}:最佳行业指数是:{}, 收益率:{}'.format(now, sector, return_index.loc[sector, 'return']))

    #if return_index.loc[sector, 'return'] < 0.005:
    #    print("所选行业收益率太低，退出选股")
    #    return

    # 获取最佳行业指数成份股
    symbols = get_history_constituents(index=sector, start_date=last_day, end_date=last_day)[0]['constituents'].keys()
    # 过滤停牌的股票
    history_instruments = get_history_instruments(symbols=symbols, start_date=now, end_date=now)
    symbols = [item['symbol'] for item in history_instruments if not item['is_suspended']]
    # 过滤退市及未上市的股票
    instrumentinfos = get_instrumentinfos(symbols=symbols, df=True)
    symbols = list(
        instrumentinfos[(instrumentinfos['listed_date'] < now) & (instrumentinfos['delisted_date'] > now)]['symbol'])
    # 获取最佳行业指数成份股的市值，选取10<市盈率<30, 5%<换手率<10% 的股票
    #fin = get_fundamentals(table='trading_derivative_indicator', symbols=symbols, start_date=last_day,
    #                       end_date=last_day, limit=context.holding_num, fields='TOTMKTCAP', order_by='-TOTMKTCAP',
    #                       df=True)
    fin = get_fundamentals(table='trading_derivative_indicator', symbols=symbols, start_date=now,
                     end_date=now, limit=context.holding_num, fields='PETTM,TURNRATE', order_by='-PETTM,-TURNRATE',
                     filter='PETTM<30 and PETTM>10 and TURNRATE>5 and TURNRATE<10', df=True)
    if len(fin) == 0:
        logger.info("行业选股无法满足要求，退出选股！")
        return
    context.to_predict = list(fin['symbol'])
    logger.info('获取符合要求的{}只股票：{}'.format(context.holding_num, context.to_predict))

    # 上一轮的交易日
    last_turn_date = get_previous_N_trading_date(last_day, counts=context.svm_training_len, exchanges='SHSE')
    # 遍历待预测股票，进行支持向量机预测
    for symbol in context.to_predict:
        # 如果仓位已满，则退出；如果仓位中已存在，则跳过
        positions = context.account().positions()
        if len(positions) >= context.holding_num:
            logger.info("仓位已满，退出预测.")
            break
        pos = [p for p in positions if p.symbol == symbol]
        if len(pos) > 0:
            logger.info("仓位中已存在该股票:{}".format(symbol))
            continue
        features = clf_fit(context, symbol, last_turn_date, last_day)
        features = np.array(features).reshape(1, -1)
        prediction = context.clf.predict(features)[0]
        # 若预测值为上涨则买入
        if prediction == 1:
            logger.info("支持向量预测买入{}".format(symbol))
            order_target_percent(symbol=symbol, percent=0.33, order_type=OrderType_Market,
                                 position_side=PositionSide_Long)



def get_previous_N_trading_date(date,counts=1,exchanges='SHSE'):
    """
    获取end_date前N个交易日,end_date为datetime格式，包括date日期
    :param date：目标日期
    :param counts：历史回溯天数，默认为1，即前一天
    """
    if isinstance(date, str) and len(date) > 10:
        date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    if isinstance(date, str) and len(date) == 10:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
    previous_N_trading_date = get_trading_dates(exchange=exchanges, start_date=date-datetime.timedelta(days=max(counts+30, counts*2)), end_date=date)[-counts]
    return previous_N_trading_date

def clf_fit(context, symbol, start_date, end_date):
    """
    训练支持向量机模型
    :param start_date:训练样本开始时间
    :param end_date:训练样本结束时间
    """
    # 获取目标股票的daily历史行情
    recent_data = history(symbol, frequency='1d', start_time=start_date, end_time=end_date, fill_missing='last',
                          df=True).set_index('eob')
    x_train = []
    y_train = []
    # 添加一行当天的数据
    curr_data = history_n(symbol, frequency='tick', count=1, end_time=context.now, fill_missing='last', df=True)
    if len(curr_data) > 0:
        dict = {"close": curr_data.iloc[0]['price'],
                      "high": curr_data.iloc[0]['high'],
                      "low": curr_data.iloc[0]['low'],
                      "volume": curr_data.iloc[0]['cum_volume']}
        df = pd.DataFrame(dict, index=[0])
        recent_data.append(df)
    # 整理训练数据
    for index in range(context.svm_history_len, len(recent_data)):
        # 自变量 X
        # 回溯N个交易日相关数据
        start_date = recent_data.index[index - context.svm_history_len]
        end_date = recent_data.index[index]
        data = recent_data.loc[start_date:end_date, :]
        # 准备训练数据
        close = data['close'].values
        max_x = data['high'].values
        min_n = data['low'].values
        volume = data['volume'].values
        close_mean = close[-1] / np.mean(close)  # 收盘价/均值
        volume_mean = volume[-1] / np.mean(volume)  # 现量/均量
        max_mean = max_x[-1] / np.mean(max_x)  # 最高价/均价
        min_mean = min_n[-1] / np.mean(min_n)  # 最低价/均价
        vol = volume[-1]  # 现量
        return_now = close[-1] / close[0]  # 区间收益率
        std = np.std(np.array(close), axis=0)  # 区间标准差
        # 将计算出的指标添加到训练集X
        x_train.append([close_mean, volume_mean, max_mean, min_mean, vol, return_now, std])

        # 因变量 Y
        if index < len(recent_data) - context.svm_forecast_len:
            y_start_date = recent_data.index[index + 1]
            y_end_date = recent_data.index[index + context.svm_forecast_len]
            y_data = recent_data.loc[y_start_date:y_end_date, 'close']
            if y_data[-1] > y_data[0]:
                label = 1
            else:
                label = 0
            y_train.append(label)

        # 最新一期的数据(返回该数据，作为待预测的数据)
        if index == len(recent_data) - 1:
            new_x_train = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
    else:
        # 剔除最后context.forecast_len期的数据
        x_train = x_train[:-context.svm_forecast_len]

    # 训练SVM
    context.clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                          tol=0.001, cache_size=200, verbose=False, max_iter=-1, decision_function_shape='ovr',
                          random_state=None)
    context.clf.fit(x_train, y_train)

    # 返回最新数据
    return new_x_train


def on_bar(context, bars):
    # 当前时间
    now = context.now
    # 获取当前时间的星期
    weekday = now.isoweekday()
    # 获取持仓
    positions = context.account().positions()
    for bar in bars:
        symbol = bar.symbol
        position = [p for p in positions if p.symbol == symbol][0]
        # 当涨幅大于10%,平掉所有仓位止盈
        if position and (now - position.created_at).days >= 1 and bar.close / position['vwap'] >= 1 + context.earn_rate:
            logger.info("平仓:{}涨幅{}>{}%,平掉仓位止盈".format(position.symbol, bar.close / position['vwap']-1, context.earn_rate*100))
            order_target_percent(symbol=position.symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
        elif position and (now - position.created_at).days >= 1 and bar.close / position['vwap'] <= 1 - context.loss_rate:
            logger.info("平仓:{}跌幅{}>{}%,平掉仓位止损".format(position.symbol, 1 - bar.close / position['vwap'], context.loss_rate*100))
            order_target_percent(symbol=position.symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
        # 当时间为周三尾盘并且涨幅小于2%时,平掉仓位止损
        #elif position and weekday == 3 and bar.close / position['vwap'] < 1 + context.sell_rate and now.hour == 14 and now.minute == 55:
        #    print("平仓:{}周三尾盘并且涨幅{}<{}%,平掉所有仓位止损".format(position.symbol, bar.close / position['vwap']-1, context.sell_rate*100))
        #    order_target_percent(symbol=position.symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
        # 当持股天数超过10天,平掉所有仓位止损
        elif position and (now - position.created_at).days >= context.holding_days and bar.close / position['vwap'] < 1 + context.sell_rate and now.hour == 14 and now.minute > 30:
            logger.info("平仓:{}持股天数超过{}并且涨幅{}<{}%，平掉所有仓位止损".format(position.symbol,context.holding_days, bar.close / position['vwap']-1, context.sell_rate*100))
            order_target_percent(symbol=position.symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)




def on_order_status(context, order):
    subFunc(context)
    # 标的代码
    symbol = order['symbol']
    # 委托价格
    price = order['price']
    # 委托数量
    volume = order['volume']
    # 目标仓位
    target_percent = order['target_percent']
    # 查看下单后的委托状态，等于3代表委托全部成交
    status = order['status']
    # 买卖方向，1为买入，2为卖出
    side = order['side']
    # 开平仓类型，1为开仓，2为平仓
    effect = order['position_effect']
    # 委托类型，1为限价委托，2为市价委托
    order_type = order['order_type']
    if status == 3:
        if effect == 1 and side == 1:
            side_effect = '开多仓'
        elif effect == 1 and side == 2:
            side_effect = '开空仓'
        elif effect == 2 and side == 1:
            side_effect = '平空仓'
        elif effect == 2 and side == 2:
            side_effect = '平多仓'
        order_type_word = '限价' if order_type == 1 else '市价'
        logger.info('{}:标的：{}，操作：以{}{}，委托价格：{}，委托数量：{}'.format(context.now, symbol, order_type_word, side_effect, price,
                                                         volume))


if __name__ == '__main__':
    '''
        strategy_id策略ID, 由系统生成
        filename文件名, 请与本文件名保持一致
        mode运行模式, 实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID, 可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式, 不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        '''
    run(strategy_id='f0c33330-b178-11ec-85b5-00ff86f797c6',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='3bb4cf8eb647a46c132ee8c6093932b873fca7c0',
        backtest_start_time='2022-01-01 08:00:00',
        backtest_end_time='2022-05-08 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)

