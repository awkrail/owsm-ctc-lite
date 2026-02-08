class TokenIDConverter:
    def __init__(
        self,
        token_list,
        unk_symbol = "<unk>",
    ):
        self.token_list = token_list
        self.token2id = {}

        for i, t in enumerate(self.token_list):
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        self.unk_id = self.token2id[self.unk_symbol]


    def get_num_vocabulary_size(self):
        return len(self.token_list)


    def ids2tokens(self, integers):
        return [self.token_list[i] for i in integers]


    def tokens2ids(self, tokens):
        return [self.token2id.get(i, self.unk_id) for i in tokens]
