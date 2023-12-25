import pandas as pd


df = pd.read_csv('comments_not_label.csv')
print('loaded')
length = len(df.comment)//5
df.iloc[length*0:length*1].to_csv('1_compiled.csv')
print('1')

df.iloc[length*1:length*2].to_csv('2_compiled.csv')
print('2')

df.iloc[length*2:length*3].to_csv('3_compiled.csv')
print('3')

df.iloc[length*3:length*4].to_csv('4_compiled.csv')
print('4')

df.iloc[length*4:length*6].to_csv('5_compiled.csv')
print('5')
