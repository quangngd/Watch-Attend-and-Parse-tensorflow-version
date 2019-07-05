export CUDA_VISIBLE_DEVICES=0

python3 model-single-GPU.py  ./data/quick/dictionary.txt \
                            ./data/quick/train.pkl \
                            ./data/quick/train.txt \
                            ./data/quick/test.pkl \
                            ./data/quick/test.txt \
                            ./data/quick/result \
                            --logPath log-quick-test-set.txt \
                            --batch_size 1 \
                            --epochSampleRatio 1 \
                            --epochValidRatio 1