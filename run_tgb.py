import os
import subprocess

tgb_test = "D:/CourtAthena/02_Datasets/TGB/dataset"

seq_list = []
if os.path.exists(tgb_test):
    seq_list = os.listdir(tgb_test)
    print(seq_list)

# python tools/demo_track.py --demo image --path D:/CourtAthena/02_Datasets/TGB/dataset/seq01/img1 -f exps/example/mot/yolox_x_mix_det.py -c pretrained/yolox_x_sports_mix.pth.tar --seq seq01
for seq in seq_list:
    print(f"Processing sequence: {seq}")
    img_path = os.path.join(tgb_test, seq, "img1")
    cmd = [
        "python", "tools/demo_track.py",
        "--demo", "image",
        "--path", img_path,
        "-f", "exps/example/mot/yolox_x_mix_det.py",
        "-c", "pretrained/yolox_x_sports_mix.pth.tar",
        "--seq", seq
    ]
    subprocess.run(cmd)