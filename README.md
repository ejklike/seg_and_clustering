# seg_and_clustering

## Modules

### Main modules
- train.py: 학습용 모듈
- interpret.py: 학습 결과 확인

### Submodules
- dataio.py: 데이터 로딩/전처리에 관한 모듈
- viz.py: 시각화 그림 저장에 관한 모듈
- utils.py: 기타 함수 모음

## Usage Statement

'''
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
'''


'''
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
'''