import sentencepiece as spm


class SentencePiecesTokenizer:
    def __init__(self, model):
        self.model = model
        self.sp = None


    def _build_sentence_piece_processor(self):
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)


    def text2tokens(self, line):
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsPieces(line)


    def token2text(self, tokens):
        self._build_sentence_piece_processor()
        return self.sp.DecodePieces(list(tokens))
