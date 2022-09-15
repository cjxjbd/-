# python train.py --dataset mpiigaze --snapshot output/snapshots --gpu 0 --num_epochs 5 --batch_size 16 --lr 0.00001 --alpha 1
# python demo_try.py --snapshot models/_epoch_5.pth.tar --gpu 0 --cam 0
# python test.py --dataset mpiigaze --snapshot output/snapshots/L2CS-mpiigaze_1661397187/fold0 --evalpath evaluation/L2CS-mpiigaze  --gpu 0

print(list('Face Left Right Origin WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin'))