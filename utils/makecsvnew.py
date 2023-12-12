import json
import csv
import os

# JSON 데이터를 파싱하는 함수
def parse_json(json_data):
    csv_data = []
    for tooth in json_data['tooth']:
        row = [
            json_data['image_filepath'].split('/')[-1],
            tooth['teeth_num'],
            tooth['segmentation'],
            tooth['decayed']
        ]
        csv_data.append(row)
    
    return csv_data

# 지정된 폴더 내의 모든 JSON 파일을 읽는 함수
def read_json_files(folder_path):
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                all_data.extend(parse_json(json_data))
    return all_data

# CSV 파일 작성
def write_to_csv(file_name, data):
    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['path', 'teeth_num', 'segmentation', 'decayed'])
        writer.writerows(data)

# JSON 폴더 경로 (이 부분을 실제 경로로 변경해야 함)
json_folder_path = '../Dataset/sample_data/json'  # 예: 'Dataset/Train_data/json'

# JSON 파일 읽기 및 데이터 추출
csv_data = read_json_files(json_folder_path)

# CSV 파일 작성
write_to_csv('output.csv', csv_data)
