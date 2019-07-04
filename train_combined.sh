export CUDA_VISIBLE_DEVICES=1

python3 model-single-GPU.py  ./combined-data/dictionary.txt \
                            ./combined-data/train.pkl \
                            ./combined-data/train.txt \
                            ./combined-data/test.pkl \
                            ./combined-data/test.txt \
                            ./combined-data/result \
                            --logPath log-combined-data-lr-1-patience-5.txt \
                            --batch_size 2 \
                            --epochSampleRatio 2 \
                            --epochValidRatio 2 \
                            --lr 1 \
                            --patience 15 \
                            --resultFileName result-combined-data-lr-1-patience-5
