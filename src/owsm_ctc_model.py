import argparse
import yaml

from audio_frontend import AudioFrontEnd
from global_mvn import GlobalMVN

class OWSM_CTC_Model:
    def __init__(
        self,
        config_path,
        model_path,
        device = "cpu",
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.device = device

        s2t_model, s2t_args = self._build_model_from_file()
        s2t_model.eval()


    def _build_model_from_file(self):
        with open(self.config_path, encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        model = self._build_model(args)


    def _build_model(self, args):
        token_list = args.token_list
        vocab_size = len(token_list)

        frontend = AudioFrontEnd()
        input_size = frontend.output_size()

        normalize = GlobalMVN(**args.normalize_conf)
        import ipdb; ipdb.set_trace()



