import torch.nn as nn


class ESPNetCTCModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        token_list,
        frontend,
        normalize,
        encoder,
        prompt_encoder,
        ctc,
        interctc_weight = 0.0,
        ignore_id = -1,
        report_cer = True,
        report_wer = True,
        sym_space = "<space>",
        sym_blank = "<blank>",
        sym_sos = "<sos>",
        sym_eos = "<eos>",
        sym_sop = "<sop>",
        sym_na = "<na>",
        extract_feats_in_collect_stats = True,
        ctc_asr_only = [False],
    ):
        super().__init__()
        
        self.blank_id = token_list.index(sym_blank)
        self.sos = token_list.index(sym_sos)
        self.eos = token_list.index(sym_eos)
        self.sop = token_list.index(sym_sop)
        self.na = token_list.index(sym_na)
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.interctc_weight = interctc_weight
        self.ctc_asr_only = ctc_asr_only
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder
        self.prompt_encoder = prompt_encoder

    
    def encode(
        self,
        speech,
        speech_lengths,
        text_prev,
        text_prev_lengths,
        prefix,
        prefix_lengths,
    ):
        feats, feats_lengths = self.frontend(speech, speech_lengths)
        feats, feats_lengths = self.normalize(feats, feats_lengths)
