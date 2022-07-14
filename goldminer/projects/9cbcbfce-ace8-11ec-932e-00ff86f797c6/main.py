# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from gm.api import *
import numpy as np
import pandas as pd

'''
行业轮动策略
逻辑：在行业指数标的中，选取历史收益最大一个行业指数
买入其成分股中最大市值的N只股票，每月月初进行调仓换股
'''

def init(context):
    # 待轮动的行业指数(分别为：300工业.300材料.300可选.300消费.300医药.300金融)
    context.index = ['SHSE.000910', 'SHSE.000909', 'SHSE.000911', 'SHSE.000912', 'SHSE.000913', 'SHSE.000914']
    # 用于统计数据的天数
    context.days = 20
    # 持股数量
    context.holding_num = 5
    # 每月定时任务（仿真和实盘时不支持该频率）
    schedule(schedule_func=algo, date_rule='1m', time_rule='09:30:00')


def algo(context):
    # 当天日期
    now = context.now
    # 获取上一个交易日
    last_day = get_previous_trading_date(exchange='SHSE', date=now)

    return_index = pd.DataFrame(columns=['return'])
    # 获取并计算行业指数收益率
    for i in context.index:
        return_index_his = history_n(symbol=i, frequency='1d', count=context.days+1, fields='close,bob',
                                     fill_missing='Last', adjust=ADJUST_PREV, end_time=last_day, df=True)
        return_index_his = return_index_his['close'].values
        return_index.loc[i,'return'] = return_index_his[-1] / return_index_his[0] - 1
    
    # 获取指定数内收益率表现最好的行业
    sector = return_index.index[np.argmax(return_index)]
    print('{}:最佳行业指数是:{}'.format(now, sector))

    # 获取最佳行业指数成份股
    symbols = get_history_constituents(index=sector, start_date=last_day, end_date=last_day)[0]['constituents'].keys()
    # 过滤停牌的股票
    history_instruments = get_history_instruments(symbols=symbols, start_date=now, end_date=now)
    symbols = [item['symbol'] for item in history_instruments if not item['is_suspended']]
    # 过滤退市及未上市的股票
    instrumentinfos = get_instrumentinfos(symbols=symbols,df=True)
    symbols = list(instrumentinfos[(instrumentinfos['listed_date']<now) & (instrumentinfos['delisted_date']>now)]['symbol'])
    # 获取最佳行业指数成份股的市值，选取市值最大的N只股票
    fin = get_fundamentals(table='trading_derivative_indicator', symbols=symbols, start_date=last_day,
                           end_date=last_day, limit=context.holding_num, fields='TOTMKTCAP', order_by='-TOTMKTCAP', df=True)
    to_buy = list(fin['symbol'])
    
    # 计算权重
    percent = 1.0 / len(to_buy)
    # 获取当前所有仓位
    positions = context.account().positions()

    # 平不在标的池的股票
    for position in positions:
        symbol = position['symbol']
        if symbol not in to_buy:
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,position_side=PositionSide_Long)

    # 买入标的池中的股票
    for symbol in to_buy:
        order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,position_side=PositionSide_Long)


def on_order_status(context, order):
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
        if effect==1 and side==1:
            side_effect = '开多仓' 
        elif effect==1 and side==2:
            side_effect = '开空仓' 
        elif effect==2 and side==1:
            side_effect = '平空仓' 
        elif effect==2 and side==2:
            side_effect = '平多仓' 
        order_type_word = '限价' if order_type==1 else '市价'
        print('{}:标的：{}，操作：以{}{}，委托价格：{}，委托数量：{}'.format(context.now,symbol,order_type_word,side_effect,price,volume))
       

def on_backtest_finished(context, indicator):
    print('*'*50)
    print('回测已完成，请通过右上角“回测历史”功能查询详情。')


if __name__ == '__main__':
    '''
    strategy_id策略ID,由系统生成
    filename文件名,请与本文件名保持一致
    mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
    token绑定计算机的ID,可在系统设置-密钥管理中生成
    backtest_start_time回测开始时间
    backtest_end_time回测结束时间
    backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
    backtest_initial_cash回测初始资金
    backtest_commission_ratio回测佣金比例
    backtest_slippage_ratio回测滑点比例
    '''
    run(strategy_id='9cbcbfce-ace8-11ec-932e-00ff86f797c6',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='3bb4cf8eb647a46c132ee8c6093932b873fca7c0',
        backtest_start_time='2019-01-01 08:00:00',
        backtest_end_time='2020-12-31 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)