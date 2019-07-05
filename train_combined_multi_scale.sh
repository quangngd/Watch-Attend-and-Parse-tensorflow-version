export CUDA_VISIBLE_DEVICES=0

python3 model-single-GPU-multi-scale.py ./combined-data/dictionary.txt \
                                        ./combined-data/train.pkl \
                                        ./combined-data/train.txt \
                                        ./combined-data/test.pkl \
                                        ./combined-data/test.txt \
                                        ./combined-data/result \
                                        --logPath log-combined-data-lr-1-patience-8-multi.txt \
                                        --batch_size 2 \
                                        --epochSampleRatio 2 \
                                        --epochValidRatio 2 \
                                        --lr 1 \
                                        --patience 8 \
                                        --resultFileName result-combined-data-lr-1-patience-8-multi