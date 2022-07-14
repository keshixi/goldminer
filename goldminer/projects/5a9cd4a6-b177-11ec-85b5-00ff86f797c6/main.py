# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from gm.api import *
import datetime

'''
本策略基于Fama-French三因子模型。
假设三因子模型可以完全解释市场，以三因子模型对每股股票进行回归计算其Alpha值，当alpha为负表明市场低估该股，因此应该买入。
策略思路：
计算市场收益率、个股的账面市值比和市值,并对后两个进行了分类,
根据分类得到的组合分别计算其市值加权收益率、SMB和HML. 
对各个股票进行回归(假设无风险收益率等于0)得到Alpha值.
选取Alpha值小于0并为最小的10只股票进入标的池，每月初移仓换股
'''


def init(context):
    # 成分股指数
    context.index_symbol = 'SHSE.000300'
    # 每月第一个交易日的09:40 定时执行algo任务（仿真和实盘时不支持该频率）
    schedule(schedule_func=algo, date_rule='1d', time_rule='15:40:00')

    # 数据滑窗
    context.date = 20

    # 设置开仓的最大资金量
    context.ratio = 0.8

    # 账面市值比的大/中/小分类
    context.BM_HIGH = 3.0
    context.BM_MIDDLE = 2.0
    context.BM_LOW = 1.0

    # 市值大/小分类
    context.MV_BIG = 2.0
    context.MV_SMALL = 1.0


def market_value_weighted(df, MV, BM):
    """
    计算市值加权下的收益率
    :param MV：MV为市值的分类对应的组别
    :param BM：BM账目市值比的分类对应的组别
    """
    select = df[(df['TOTMKTCAP'] == MV) & (df['BM'] == BM)] # 选出市值为MV，账目市值比为BM的所有股票数据
    mv_weighted = select['mv']/np.sum(select['mv'])# 市值加权的权重
    return_weighted = select['return']*mv_weighted# 市值加权下的收益率
    return np.sum(return_weighted)
    

def algo(context):
    # 当前时间
    now = context.now
    fin = get_fundamentals(table='trading_derivative_indicator', symbols=['SHSE.600000', 'SZSE.000001'], start_date=now,
                           end_date=now, limit=5, fields='PETTM,TURNRATE', order_by='-PETTM,-TURNRATE',
                           filter='',
                           df=True)
    print(fin)
    print(fin.iloc[0]['PETTM'])
    print(fin.iloc[0]['TURNRATE'])
    #predate = get_trading_dates(exchange='SHSE', start_date=now - datetime.timedelta(days=max(1 + 30, 1 * 2)),
    #                  end_date=now)[-1]
   # recent_data = history_n('SHSE.600649', frequency='tick',count=1, end_time=now, fill_missing='last',
    #                      df=True)
    #print(len(recent_data))
    #print(recent_data)
    #print(recent_data.iloc[0])
    #print(recent_data.iloc[0]['open'])
   # print(recent_data.iloc[0]['close'])
    #print(recent_data.iloc[0]['high'])
    #print(recent_data.iloc[0]['low'])
    #print(recent_data.iloc[0]['volume'])



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
        print('{}:标的：{}，操作：以{}{}，委托价格：{}，目标仓位：{:.2%}'.format(context.now,symbol,order_type_word,side_effect,price,target_percent))


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
    run(strategy_id='5a9cd4a6-b177-11ec-85b5-00ff86f797c6',
        filename='main.py',
        mode=MODE_LIVE,
        token='3bb4cf8eb647a46c132ee8c6093932b873fca7c0',
        backtest_start_time='2022-04-29 08:00:00',
        backtest_end_time='2022-04-29 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)