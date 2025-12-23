import os
import subprocess

WITH_GT = False

trackid3x3_test = "D:/CourtAthena/02_Datasets/trackid3x3_720p/test"

seq_list = []
if os.path.exists(trackid3x3_test):
    seq_list = os.listdir(trackid3x3_test)
    print(seq_list)

# python tools/demo_track.py --demo image --path D:/CourtAthena/02_Datasets/trackid3x3_720p/test/IMG_0106/img1 -f exps/example/mot/yolox_x_mix_det.py -c pretrained/yolox_x_sports_mix.pth.tar --seq IMG_0106
# python tools/demo_track.py --demo image --path D:/CourtAthena/02_Datasets/trackid3x3_720p/test/IMG_0106/img1 -f exps/example/mot/yolox_x_mix_det.py -c pretrained/yolox_x_sports_mix.pth.tar --seq IMG_0106 --det_path D:/CourtAthena/02_Datasets/trackid3x3_720p/test/IMG_0106/gt/gt.txt
for seq in seq_list:
    print(f"Processing sequence: {seq}")
    img_path = os.path.join(trackid3x3_test, seq, "img1")
    gt_path = os.path.join(trackid3x3_test, seq, "gt", "gt.txt")

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