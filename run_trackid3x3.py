import os
import subprocess

trackid3x3_test = "D:/CourtAthena/02_Datasets/trackid3x3_720p/dataset"

seq_list = []
if os.path.exists(trackid3x3_test):
    seq_list = os.listdir(trackid3x3_test)
    print(seq_list)

# python tools/demo_track.py --demo image --path D:/CourtAthena/02_Datasets/trackid3x3_720p/dataset/IMG_0104/img1 -f exps/example/mot/yolox_x_mix_det.py -c pretrained/yolox_x_sports_mix.pth.tar --seq IMG_0104
for seq in seq_list:
    print(f"Processing sequence: {seq}")
    img_path = os.path.join(trackid3x3_test, seq, "img1")
    cmd = [
        "python", "tools/demo_track.py",
        "--demo", "image",
        "--path", img_path,
        "-f", "exps/example/mot/yolox_x_mix_det.py",
        "-c", "pretrained/yolox_x_sports_mix.pth.tar",
        "--seq", seq
    ]
    subprocess.run(cmd)