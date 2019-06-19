# seg_and_clustering

## Modules

### Main modules
- `train.py`: 학습용 모듈
- `interpret.py`: 학습 결과 확인

### Submodules
- `dataio.py`: 데이터(★) 로딩/전처리에 관한 모듈
  - 새로운 데이터에 적용하고 싶다면 이 모듈 내 함수 및 DB쿼리 부분을 수정할 것.
- `graph.py`: 클러스터 별 센서 네트워크 그래프 시각화 모듈
- `viz.py`: 구간 분할 결과 시각화 모듈
- `utils.py`: 기타 함수 모음
 

## Usage Statements and Examples

### 학습: `train.py`

- 원하는 클러스터 개수 범위(`min_nc`~`max_nc`)에 대해 모델 학습 및 결과 `pkl`파일 저장
- 실험 결과, 클러스터 개수에 따른 log-likelihood, BIC 결과를 한 데 모은 시각화 plot이 저장되므로, 이를 살펴 최적 클러스터 개수를 선택
- 최적 클러스터 모델에 대한 상세한 결과는 `interpret.py`로 확인

- Usage examples

```
python train.py --min_nc 3 --max_nc 3 --ws 1
python train.py --verbose --min_nc 3 --max_nc 10 --maxiter 10 --ws 2
python train.py --verbose --test_size 0 --min_nc 3 --max_nc 10 --maxiter 10 --ws 2
```

- Usage statement
```
> python train.py -h
usage: train.py [-h] [--verbose [VERBOSE]] [--num_proc NUM_PROC] [--ld LD]
                [--bt BT] [--ws WS] [--maxiter MAXITER]
                [--threshold THRESHOLD] [--min_nc MIN_NC] [--max_nc MAX_NC]
                [--test-size TEST_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --verbose [VERBOSE]   verbose for TICC
  --num_proc NUM_PROC   the number of threads
  --ld LD               lambda (sparsity in Toplitz matrix)
  --bt BT               beta (segmentation penalty)
  --ws WS               window size
  --maxiter MAXITER     maxiter
  --threshold THRESHOLD
                        threshold
  --min_nc MIN_NC       min_num_cluster
  --max_nc MAX_NC       max_num_cluster
  --test-size TEST_SIZE
                        test data size
```

### 모델 살피기: `interpret.py`

- `train.py` 실행 결과 저장된 모델을 자세히 확인하기 위한 모듈
- 클러스터 별 센서 네트워크 구조 시각화 및 저장
- 학습/테스트 데이터에 대한 구간 분할 적용 결과 시각화 및 저장

- Usage examples
  - 원하는 클러스터 개수를 입력해주어야 함.
  - `train.py`에서 사용한 파라미터 설정을 그대로 입력해주어야 함.
  - 아래 예시는 위에 `train.py` 예시 코드를 실행하여 도출된 BIC 점수를 살펴본 결과 각각의 예시에서 최적 클러스터 개수가 `3`, `8`, `6`인 상황을 가정하였음

```
python interpret.py 3 --ws 1
python interpret.py 8 --maxiter 10 --ws 2
python interpret.py 6 --test_size 0 --ws 2
```

- Usage statements

```
> python interpret.py -h
usage: interpret.py [-h] [--ld LD] [--bt BT] [--ws WS] [--threshold THRESHOLD]
                    [--test-size TEST_SIZE]
                    nc

positional arguments:
  nc                    num_cluster to interpret

optional arguments:
  -h, --help            show this help message and exit
  --ld LD               lambda (sparsity in Toplitz matrix)
  --bt BT               beta (segmentation penalty)
  --ws WS               window size
  --threshold THRESHOLD
                        threshold
  --test-size TEST_SIZE
                        test data size
```