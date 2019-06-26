export CUDA_VISIBLE_DEVICES=0

python3 model-single-GPU-multi-scale.py     ./data/quick/dictionary.txt \
                                            ./data/quick/train.pkl \
                                            ./data/quick/train.txt \
                                            ./data/quick/test.pkl \
                                            ./data/quick/test.txt \
                                            ./data/quick/result \
                                            --logPath log-test.txt \
                                            --batch_size 2 \
                                            --epochSampleRatio 1 \
                                            --epochValidRatio 1