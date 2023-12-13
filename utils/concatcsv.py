import pandas as pd
import os

# CSV 파일 목록 (이 부분을 실제 파일 경로로 변경해야 함)
csv_files = [
    'frontinput.csv',
    'leftinput.csv',
    'lowerinput.csv',
    'rightinput.csv',
    'upperinput.csv'
]

# 모든 CSV 파일을 하나의 DataFrame으로 읽기
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# 합쳐진 DataFrame을 새로운 CSV 파일로 저장
combined_df.to_csv('test_combined_file.csv', index=False)
