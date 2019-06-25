export CUDA_VISIBLE_DEVICES=0

python3 model-single-GPU.py  ./combined-data/dictionary.txt \
                            ./combined-data/train.pkl \
                            ./combined-data/train.txt \
                            ./combined-data/test.pkl \
                            ./combined-data/test.txt \
                            ./combined-data/result \
                            --logPath log-combined-data.txt \
                            --batch_size 2 \
                            --epochSampleRatio 4 \
                            --epochValidRatio 4