# 디렉토리 경로
directory_path="/jsphilip54/Dataset/train_data/imagecrop"


while true; do
    # 디렉토리 내 파일 개수 확인
    file_count=$(ls -1 "$directory_path" | wc -l)

    # 파일 개수가 10보다 큰 경우에는 script1.py를 실행
    if [ "$file_count" -lt 63640 ]; then
        pgrep -f croptoothlower.py || nohup python croptoothlower.py
    # 파일 개수가 20보다 큰 경우에는 script2.py를 실행
    elif [ "$file_count" -lt 126882 ]; then
        pgrep -f croptoothupper.py || nohup python croptoothupper.py
    # 파일 개수가 30보다 큰 경우에는 script3.py를 실행
    elif [ "$file_count" -lt 189965 ]; then
        pgrep -f croptoothleft.py || nohup python croptoothleft.py
    # 파일 개수가 40보다 큰 경우에는 script4.py를 실행
    elif [ "$file_count" -lt 275616 ]; then
        pgrep -f croptoothfront.py || nohup python croptoothfront.py
    # 파일 개수가 50보다 큰 경우에는 script5.py를 실행
    elif [ "$file_count" -lt 338719 ]; then
        pgrep -f croptoothright.py || nohup python croptoothright.py
    fi

    # 파일 개수를 주기적으로 확인하기 위한 대기 시간 설정 (예: 1분)
    sleep 60
done