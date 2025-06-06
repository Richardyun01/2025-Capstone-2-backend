# 백엔드 테스트
1. uvicorn api_server:app --reload 실행
2. http://127.0.0.1:8000/docs/ 접속

# backend: 삼변측량
## request
- GET

## path
- /get_position

## query parameter
- up_distance: float
- down_distance: float
- front_distance: float
- left_distance: float
- arm_length: float

## response
```
{
  "x": float,
  "y": float,
  "z": float
}
```

# backend: PCAP features
## 학습-테스트
1. capture 폴더에 분류 별로 파일 준비
2. (선택) stratified_split.py로 test에 사용할 파일 비율 확인
3. batch_extract.py로 모든 pcap 파일을 추출
4. 생성된 feature와 label을 train과 test용으로 파일 분리
5. load_and_test_dataset.py를 실행해서 학습 및 평가

## 학습 완료 후 테스트
1. evaluation.py를 실행해서 평가

## 단일 pcap 파일 검사
1. python load_and_test_dataset.py --test-pcap your_capture.pcap

## request
- GET

## path
- /pcap/analyze

## query parameter
- file: File

## response
```
{
  "targets": [ { List [dict] } ]
}
```
- 리스트 내의 딕셔너리 src_mac, dest_mac
- ex
  ```
  {
    "targets": [
      { "src_mac": "02:00:11:22:75:14", "dest_mac": "56:8c:94:88:e5:0d" },
      { "src_mac": "3c:67:a2:38:4c:98", "dest_mac": "9c:25:95:3c:bc:0f" }
    ]
  }
  ```
