from layer import *


class SSM(nn.Module):

    def __init__(self, poi_num, long_seq_length, user_emb_num, poi_emb_num, dis_emb_num, args):
        super(SSM, self).__init__()
        # parameters
        self.poi_num = poi_num
        self.in_dim = args.emb_dim
        self.mid_dim = args.mid_dim
        self.out_dim = args.emb_dim
        self.emb_dim = args.emb_dim
        self.drop_long = args.drop[0]
        self.drop_short = args.drop[1]
        self.drop_fusion = args.drop[2]
        self.head_number = args.head_number
        self.long_seq_length = long_seq_length
        self.short_seq_length = args.short_length
        self.iterations_caps = args.iteration_caps
        self.extractor_number = args.extractor_number
        self.low_capsule_num = self.head_number*2
        # initialize embed layer
        self.embedding_user = nn.Embedding(num_embeddings=user_emb_num, embedding_dim=self.emb_dim)
        self.embedding_poi = nn.Embedding(num_embeddings=poi_emb_num, embedding_dim=self.emb_dim, padding_idx=0)
        self.embedding_time = nn.Embedding(num_embeddings=int(24/args.time_size)+1, embedding_dim=self.emb_dim, padding_idx=0)
        self.embedding_interval = nn.Embedding(num_embeddings=25, embedding_dim=self.emb_dim)
        self.embedding_distance = nn.Embedding(num_embeddings=dis_emb_num, embedding_dim=self.emb_dim)
        # initialize long interest model
        self.long_u2p = ALE(self.emb_dim, self.head_number, self.extractor_number, self.drop_long)
        self.long_t2p = ALE(self.emb_dim, self.head_number, self.extractor_number, self.drop_long)
        self.long_interest = nn.Linear(self.emb_dim * 2, self.emb_dim)
        # initialize short interest model
        self.short = ASE(self.in_dim, self.mid_dim, self.out_dim, self.drop_short)
        self.short_interest = MultiHeadAttention(self.emb_dim, self.head_number)
        # initialize the fusion recommendation model
        self.fusion = Capsule(self.emb_dim, self.poi_num, self.iterations_caps, self.low_capsule_num, self.drop_fusion)

    def forward(self, long_user, long_poi, long_mask, long_time, short_user, short_poi, short_interval, short_distance):

        # embedding
        long_user_ed = self.embedding_user(long_user)
        long_poi_ed = self.embedding_poi(long_poi)
        long_time_ed = self.embedding_time(long_time)
        short_user_ed = self.embedding_user(short_user[:, :1])
        short_poi_ed = self.embedding_poi(short_poi)
        short_interval_ed = self.embedding_interval(short_interval)
        short_distance_ed = self.embedding_distance(short_distance)

        # model the long interest
        out_long_u2p = self.long_u2p(long_user_ed, long_poi_ed, long_mask)
        out_long_t2p = self.long_t2p(long_time_ed, long_poi_ed, long_mask)
        long_interest = squash(self.long_interest(torch.cat((out_long_u2p, out_long_t2p), dim=-1)))

        # model the short interest
        short_list = self.short(short_poi_ed, short_distance_ed, short_interval_ed)  # list
        short_cat = torch.cat(short_list, dim=1)
        short_interest = self.short_interest(short_user_ed, short_cat)

        # for matching shape
        # if short_interest.shape[1] < self.short_seq_length:
        #     cus, yus = divmod(self.short_seq_length, short_interest.shape[1])
        #     repeat_num = torch.zeros((short_interest.shape[1],), dtype=torch.long).to(short_interest.device)
        #     repeat_num[:] = cus
        #     repeat_num[:yus] += 1
        #     short_interest = torch.repeat_interleave(short_interest, repeat_num, dim=1)

        # together predict
        low_caps = torch.cat((long_interest, short_interest), dim=1)
        out = self.fusion(low_caps)

        return out
