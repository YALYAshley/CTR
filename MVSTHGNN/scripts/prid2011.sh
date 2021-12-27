#!/bin/bash
for i in 0
do
python /home/mn/桌面/STHGNN/train_vidreid_xent_htri.py -d prid2011 \
                              --root ./data \
                              --seq-len 8 \
                              --train-batch 8 \
                              --test-batch 16 \
                              --num-instances 4 \
                              --train-sample restricted \
                              --test-sample restricted \
                              --train-sampler RandomIdentitySamplerV1 \
                              --optim adam \
                              --soft-margin \
                              --label-smooth \
                              --warmup \
                              --misalign-aug \
                              --max-epoch 300 \
                              --lr 1e-4 \
                              --stepsize 100 200 300 \
                              -a vmgn_hgnn \
                              --pyramid-part \
                              --num-gb 2 \
                              --learn-graph \
                              --print-last \
                              --gpu-devices 0 \
                              --eval-step 5 \
                              --start-eval 50 \
                              --dist-metric cosine \
                              --is-probH \
                              --k-neigs 3 \
                              --global-k-neigs 5 \
                              --m-prob 1.0 \
                              --drop-out 0.5 \
                              --crop-scale-h 0.5 \
                              --crop-scale-w 0.25 \
                              --use-pose \
                              --mode all_graph \
                              --node-num 6 6 6 \
                              --split-id $i \
                              --global-branch \
                              --learn-edge \
                              --save-dir  log/prid2011-ngb2-all_graph
done
