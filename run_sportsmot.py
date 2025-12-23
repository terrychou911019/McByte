import os
import subprocess

WITH_GT = False

sportsmot_val = "D:/CourtAthena/02_Datasets/sportsmot_publish/dataset/val"

seq_list = []
if os.path.exists(sportsmot_val):
    seq_list = os.listdir(sportsmot_val)
    print(seq_list)

# python tools/demo_track.py --demo image --path D:/CourtAthena/02_Datasets/sportsmot_publish/dataset/val/v_00HRwkvvjtQ_c001/img1 -f exps/example/mot/yolox_x_mix_det.py -c pretrained/yolox_x_sports_mix.pth.tar --seq v_00HRwkvvjtQ_c001
# python tools/demo_track.py --demo image --path D:/CourtAthena/02_Datasets/sportsmot_publish/dataset/val/v_00HRwkvvjtQ_c001/img1 -f exps/example/mot/yolox_x_mix_det.py -c pretrained/yolox_x_sports_mix.pth.tar --seq v_00HRwkvvjtQ_c001 --det_path D:/CourtAthena/02_Datasets/sportsmot_publish/dataset/val/v_00HRwkvvjtQ_c001/gt/gt.txt
for seq in seq_list:
    print(f"Processing sequence: {seq}")
    img_path = os.path.join(sportsmot_val, seq, "img1")
    gt_path = os.path.join(sportsmot_val, seq, "gt", "gt.txt")

    if WITH_GT:
        cmd = [
            "python", "tools/demo_track.py",
            "--demo", "image",
            "--path", img_path,
            "-f", "exps/example/mot/yolox_x_mix_det.py",
            "-c", "pretrained/yolox_x_sports_mix.pth.tar",
            "--seq", seq,
            "--det_path", gt_path
        ]
    else:
        cmd = [
            "python", "tools/demo_track.py",
            "--demo", "image",
            "--path", img_path,
            "-f", "exps/example/mot/yolox_x_mix_det.py",
            "-c", "pretrained/yolox_x_sports_mix.pth.tar",
            "--seq", seq
        ]
        
    subprocess.run(cmd)