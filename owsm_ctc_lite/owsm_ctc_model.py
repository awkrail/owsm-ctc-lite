import argparse
import yaml
import torch
import librosa

from owsm_ctc_lite.audio_frontend import AudioFrontEnd
from owsm_ctc_lite.global_mvn import GlobalMVN
from owsm_ctc_lite.e_branchformer_ctc_encoder import EBranchformerCTCEncoder
from owsm_ctc_lite.transformer_encoder import TransformerEncoder
from owsm_ctc_lite.ctc import CTC
from owsm_ctc_lite.espnet_ctc_model import ESPNetCTCModel
from owsm_ctc_lite.tokenizer import SentencePiecesTokenizer
from owsm_ctc_lite.token_id_converter import TokenIDConverter
from owsm_ctc_lite.utils import pad_list, to_device_dict


class OWSMCTCModel:
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
        frames_per_sec = self.sample_rate / s2t_args.frontend_conf["hop_length"]
        frames_per_sec /= subsample_factor
        self.frames_per_sec = frames_per_sec


    def predict(
        self,
        audio_path,
        batch_size = 1, # inference - single file only
        context_len_in_secs = 4,
        text_prev = "<na>",
    ):
        (
            all_speech,
            all_speech_lengths,
            all_text_prev,
            all_text_prev_lengths,
            all_prefix,
            all_prefix_lengths,
            n_chunks,
        ) = self._prepare_inputs(audio_path, context_len_in_secs, text_prev)
        
        batch = {
            "speech" : all_speech,
            "speech_lengths" : all_speech_lengths,
            "text_prev" : all_text_prev,
            "text_prev_lengths" : all_text_prev_lengths,
            "prefix" : all_prefix,
            "prefix_lengths" : all_prefix_lengths,
        }
        batch = to_device_dict(batch, device=self.device)
        enc, enc_olens = self.s2t_model.encode(**batch)


    def _prepare_inputs(
        self,
        speech,
        context_len_in_secs,
        text_prev,
    ):
        sample_rate = self.sample_rate
        lang_sym = self.lang_sym
        task_sym = self.task_sym

        buffer_len_in_secs = self.s2t_args.preprocessor_conf["speech_length"]
        chunk_len_in_secs = buffer_len_in_secs - 2 * context_len_in_secs
        buffer_len = int(sample_rate * buffer_len_in_secs)
        chunk_len = int(sample_rate * chunk_len_in_secs)

        n_chunks = []
        all_speech = []
        all_text_prev = []
        all_prefix = []

        speech, _ = librosa.load(speech, sr=sample_rate)
        if len(speech) <= buffer_len:
            n_chunks.append(1)
            cur_speech = librosa.util.fix_length(speech, size=buffer_len)
            all_speech.append(
                torch.tensor(cur_speech, dtype=getattr(torch, self.dtype))
            )

            cur_text_prev = self.converter.tokens2ids(
                self.tokenizer.text2tokens(text_prev)
            )

            if self.s2t_model.na in cur_text_prev:
                cur_text_prev = [self.s2t_model.na]
            all_text_prev.append(torch.tensor(cur_text_prev, dtype=torch.long))

            lang_id = self.converter.token2id[lang_sym]
            task_id = self.converter.token2id[task_sym]

            all_prefix.append(
                torch.tensor([lang_id, task_id], dtype=torch.long)
            )

        else:
            raise NotImplementedError() # TODO: implement >30s audio

        all_speech = torch.stack(all_speech)
        all_speech_lengths = all_speech.new_full(
            [all_speech.size(0)], dtype=torch.long, fill_value=all_speech.size(1),
        )

        all_text_prev_lengths = torch.tensor(
            [x.size(0) for x in all_text_prev], dtype=torch.long
        )
        all_text_prev = pad_list(all_text_prev, self.s2t_model.eos)

        all_prefix = torch.stack(all_prefix)
        all_prefix_lengths = all_prefix.new_full(
            [all_prefix.size(0)], dtype=torch.long, fill_value=all_prefix.size(1),
        )

        return (
            all_speech,
            all_speech_lengths,
            all_text_prev,
            all_text_prev_lengths,
            all_prefix,
            all_prefix_lengths,
            n_chunks,
        )


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
        frontend = AudioFrontEnd(**args.frontend_conf)
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
