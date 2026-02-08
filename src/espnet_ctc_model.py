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
        self.token_list = token_list
