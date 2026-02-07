import argparse
import yaml

from audio_frontend import AudioFrontEnd
from global_mvn import GlobalMVN
from e_branchformer_ctc_encoder import EBranchformerCTCEncoder
from transformer_encoder import TransformerEncoder
from ctc import CTC
from espnet_ctc_model import ESPNetCTCModel

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
        import ipdb; ipdb.set_trace()


    def _build_model(self, args):
        token_list = args.token_list
        vocab_size = len(token_list)

        # Audio frontend
        frontend = AudioFrontEnd()
        input_size = frontend.output_size()

        # Normalizer
        normalize = GlobalMVN(**args.normalize_conf)

        # Encoder
        encoder = EBranchformerCTCEncoder(input_size = input_size, **args.encoder_conf)

        # PromptEncoder
        prompt_encoder = TransformerEncoder(
            input_size = args.promptencoder_conf["output_size"],
            input_layer = None,
            **args.promptencoder_conf,
        )

        # CTC layer
        ctc = CTC(
            odim = vocab_size, encoder_output_size = encoder.output_size(), **args.ctc_conf
        )

        model = ESPNetCTCModel(
            vocab_size = vocab_size,
            frontend = frontend,
            normalize = normalize,
            encoder = encoder,
            prompt_encoder = prompt_encoder,
            ctc = ctc,
            token_list = token_list,
            **args.model_conf,
        )

        return model
