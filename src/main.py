import argparse
from owsm_ctc_model import OWSM_CTC_Model
# from audio_preprocessor import AudioPreprocessor


def main(args):
    model = OWSM_CTC_Model(
        config_path = "model_card/config.yaml",
        model_path = "model_card/valid.total_count.ave_5best.till40epoch.pth",
    )
    res = model.predict(args.input_audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio_path", "-i", default="jfk.wav", help="input audio path")
    args = parser.parse_args()
    main(args)
