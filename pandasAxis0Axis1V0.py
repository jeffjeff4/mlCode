####记忆技巧
####axis=0:
####
####想象表格从上到下"压扁"(垂直方向)
####
####操作会减少行数或跨列计算
####
####对应列方向的操作
####
####axis=1:
####
####想象表格从左到右"压扁"(水平方向)
####
####操作会减少列数或跨行计算
####
####对应行方向的操作
####
####实际应用场景
####使用 axis=0 的场景：
####
####计算每列的平均值、总和等统计量
####
####删除特定行
####
####垂直合并多个DataFrame
####
####使用 axis=1 的场景：
####
####计算每行的统计量
####
####删除特定列
####
####水平合并多个DataFrame
####
####跨多列应用函数
####
####注意事项
####某些函数默认使用 axis=0 (如 sum(), mean())
####
####在 drop() 方法中，也可以使用 axis='index' 或 axis='columns' 来替代数字
####
####在多层索引(MultiIndex)的情况下，axis 的行为可能会更复杂
##
##import pandas as pd
##import numpy as np
##
### 创建示例DataFrame
##df = pd.DataFrame({
##    'A': [1, 2, 3],
##    'B': [4, 5, 6],
##    'C': [7, 8, 9]
##}, index=['row1', 'row2', 'row3'])
##
##print("原始DataFrame:")
##print(df)
##
### 对每列求和 (垂直方向)
##col_sum = df.sum(axis=0)
##print("\n对每列求和 (axis=0):")
##print(col_sum)
##
### 对每行求和 (水平方向)
##row_sum = df.sum(axis=1)
##print("\n对每行求和 (axis=1):")
##print(row_sum)
##
### 删除行 (axis=0 或 axis='index')
##df_drop_row = df.drop('row1', axis=0)
##print("\n删除行 (axis=0):")
##print(df_drop_row)
##
### 删除列 (axis=1 或 axis='columns')
##df_drop_col = df.drop('A', axis=1)
##print("\n删除列 (axis=1):")
##print(df_drop_col)
##
### 对每列应用函数
##col_max = df.apply(lambda x: x.max(), axis=0)
##print("\n每列的最大值 (axis=0):")
##print(col_max)
##
### 对每行应用函数
##row_mean = df.apply(lambda x: x.mean(), axis=1)
##print("\n每行的平均值 (axis=1):")
##print(row_mean)
##
### 创建第二个DataFrame
##df2 = pd.DataFrame({
##    'A': [10, 11],
##    'B': [12, 13],
##    'C': [14, 15]
##}, index=['row4', 'row5'])
##
### 垂直连接 (增加行)
##df_concat_0 = pd.concat([df, df2], axis=0)
##print("\n垂直连接 (axis=0):")
##print(df_concat_0)
##
##
### 创建第三个DataFrame (相同行索引)
##df3 = pd.DataFrame({
##    'D': [10, 11, 12],
##    'E': [13, 14, 15]
##}, index=['row1', 'row2', 'row3'])
##
### 水平连接 (增加列)
##df_concat_1 = pd.concat([df, df3], axis=1)
##print("\n水平连接 (axis=1):")
##print(df_concat_1)


