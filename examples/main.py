import argparse
import torch
import sys
sys.path.append(".")

from owsm_ctc_lite.owsm_ctc_model import OWSMCTCModel


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OWSMCTCModel(
        config_path = "model_card/config.yaml",
        model_path = "model_card/valid.total_count.ave_5best.till40epoch.pth",
        device = device,
    )
    res = model.predict(args.input_audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio_path", "-i", default="jfk.wav", help="input audio path")
    args = parser.parse_args()
    main(args)
