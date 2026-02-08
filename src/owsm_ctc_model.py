import argparse
import yaml
import torch

from audio_frontend import AudioFrontEnd
from global_mvn import GlobalMVN
from e_branchformer_ctc_encoder import EBranchformerCTCEncoder
from transformer_encoder import TransformerEncoder
from ctc import CTC
from espnet_ctc_model import ESPNetCTCModel
from tokenizer import SentencePiecesTokenizer
from token_id_converter import TokenIDConverter


class OWSM_CTC_Model:
    def __init__(
        self,
        config_path,
        model_path,
        device = "cpu",
        batch_size = 1,
        dtype = "float32",
        lang_sym = "<nolang>",
        task_sym = "<asr>",
        use_flash_attn = False,
        generate_interctc_outputs = False,
        **kwargs,
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.device = device

        # model and args
        s2t_model, s2t_args = self._build_model_from_file()
        s2t_model.eval()

        # set flash_attn
        for m in s2t_model.modules():
            if hasattr(m, "use_flash_attn"):
                setattr(m, "use_flash_attn", use_flash_attn)

        # tokenizer
        bpemodel = s2t_args.bpemodel
        tokenizer = SentencePiecesTokenizer(s2t_args.bpemodel)
        converter = TokenIDConverter(s2t_model.token_list)

        self.s2t_model = s2t_model
        self.s2t_args = s2t_args
        self.preprocessor_conf = s2t_args.preprocessor_conf
        self.converter = converter
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.generate_interctc_outputs = generate_interctc_outputs

        # default symbols
        self.lang_sym = lang_sym
        self.task_sym = task_sym

        # sample rate and fps
        self.sample_rate = s2t_args.frontend_conf["fs"]
        subsample_dict = {
            "conv2d1" : 1,
            "conv2d2" : 2,
            "conv2d" : 4,
            "conv2d6" : 6,
            "conv2d8" : 8,
        }
        subsample_factor = subsample_dict[s2t_args.encoder_conf["input_layer"]]
        frames_per_sec = sample_rate / s2t_args.frontend_conf["hop_length"]
        frames_per_sec /= subsample_factor
        self.frames_per_sec = frames_per_sec


    def predict(self):
        import ipdb; ipdb.set_trace()
        pass


    def _build_model_from_file(self):
        # prepare config & model
        with open(self.config_path, encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        model = self._build_model(args)
        model.to(self.device)

        # load model
        state_dict = torch.load(self.model_path, weights_only = False)
        model.load_state_dict(state_dict, strict=False)
        return model, args


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
