# backend: 삼변측량

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

## 백엔드 테스트
1. uvicorn api_server:app --reload 실행
2. http://127.0.0.1:8000/docs/ 접속
3. Try it out 선택 -> 파일 업로드 후 실행

## 엔드포인트
- GET: /
  - 서버 상태 확인
- POST: /pcap/analyze
  - 구동 엔드포인트

## 출력 JSON 형식
```
{
  "targets": [
    "02:00:11:22:92:f7|56:8c:94:88:e5:0d"
  ]
}
```