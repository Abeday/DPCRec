import argparse
from model import *
from prepare import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset


seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

config = argparse.ArgumentParser()

# model parameter
config.add_argument('--emb_dim', type=int, default=64)
config.add_argument('--mid_dim', type=int, default=128)  # gru
config.add_argument('--time_size', type=int, default=3)  # the size of time-boxing
config.add_argument('--head_number', type=int, default=48)  # half of low caps num
config.add_argument('--short_length', type=int, default=24)
config.add_argument('--num_negative', type=int, default=10)
config.add_argument('--iteration_caps', type=int, default=6)  # capsule
config.add_argument('--extractor_number', type=int, default=10)
cut_limit = 0
evaluation_size = 1
# train parameter
config.add_argument('--num_epoch', type=int, default=100)
config.add_argument('--batch_size', type=int, default=23)
config.add_argument('--learning_rate', type=float, default=1e-4)
config.add_argument('--drop', type=float, nargs='+', default=[0.2, 0.2, 0])  # rate:[long, short, fusion]

config.add_argument('--dataset', type=str, default='nyc')
config.add_argument('--device', type=str, default='cuda')

arguments = config.parse_args()


class SsmDataSet(Dataset):

    def __init__(self, user_seq, poi_seq, mask, time_seq, interval_seq, distance_seq, device):

        self.user_seq = user_seq
        self.poi_seq = poi_seq
        self.mask = mask
        self.time_seq = time_seq
        self.interval_seq = interval_seq
        self.distance_seq = distance_seq
        self.device = device

    def __getitem__(self, index):

        user_seg = self.user_seq[index].to(self.device)
        long_seg = self.poi_seq[index].to(self.device)
        mask = self.mask.to(self.device)
        time_seg = self.time_seq[index].to(self.device)
        interval_seg = self.interval_seq[index].to(self.device)
        distance_seg = self.distance_seq[index].to(self.device)

        return user_seg, long_seg, mask, time_seg, interval_seg, distance_seg

    def __len__(self):
        return self.user_seq.shape[0]


class ModelManager(object):

    def __init__(self, poi_num, long_seq_length, user_emb_num, poi_emb_num, dis_emb_num, args):

        self.eva_size = evaluation_size
        self.poi_num = poi_num
        self.device = args.device
        self.bs = args.batch_size
        self.nep = args.num_epoch
        self.lr = args.learning_rate
        self.neg = args.num_negative
        self.short_len = args.short_length
        self.criterion = nn.CrossEntropyLoss()
        self.model = SSM(poi_num, long_seq_length, user_emb_num, poi_emb_num, dis_emb_num, args).to(self.device)

        self.threshold = 0
        self.save_index_val = []
        self.save_index_tes = []
        self.save_epoch = []
        self.save_acc_val = []
        self.save_acc_tes = []
        self.record = {'epoch': [], 'valid_ac': [], 'test_ac': []}
        self.model_path = 'data/' + arguments.dataset + '/model_ssm.pth'

    def run(self, user_seq, poi_seq, mask, time_seq, interval_seq, distance_seq):

        assert user_seq.shape[0] % self.bs == 0, \
            f'!!!{user_seq.shape[0]} should be divisible by the batch size{self.bs}!!!'
        operator_seq_len = user_seq.shape[1] - 1
        dataset = SsmDataSet(user_seq, poi_seq, mask, time_seq, interval_seq, distance_seq, self.device)
        loader = DataLoader(dataset, batch_size=self.bs, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=33, gamma=0.5)

        new_tar = torch.zeros((self.bs, self.poi_num)).to(self.device)
        woyaonulecaocaocao = torch.zeros((len(dataset)//self.bs,)).to(self.device)
        wocaonima = []
        for epoch in range(self.nep):  # epoch start
            woyaonulecaocaocao.fill_(0)
            wocaonima.clear()
            acc_tra, acc_val, acc_tes = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
            mrr_tra, mrr_val, mrr_tes = 0, 0, 0
            train_size, valid_size, test_size = 0, 0, 0
            gap_train_valid = self.eva_size*2
            gap_valid_test = self.eva_size
            loss_show = 0.0
            s, e = 0, 0

            global seed
            random.seed(seed)
            data_index = list(range(operator_seq_len))

            bar = tqdm(total=len(dataset) / self.bs * operator_seq_len, desc=f'--epoch{epoch+1}', ncols=90)
            for i_batch, (user, poi, mask, time, itv, dist) in enumerate(loader):
                # mask(len, len, len) just used in long interest model
                # random.shuffle(data_index)  # data_index : each step is different but each epoch is same

                old_acc = acc_tes[0]
                for i, d_i in enumerate(data_index):
                    new_tar.fill_(0)  # init target every step

                    s = max(0, d_i - self.short_len + 1)  # start index of the short interest model input
                    e = d_i + 1  # end index of the short interest model input

                    for b in range(self.bs):  # build target
                        new_tar[b, poi[b, d_i + 1] - 1] = 1

                    # train
                    if i < len(data_index)-gap_train_valid:
                        self.model.train()
                        train_size += self.bs

                        prob_train = self.model(user, poi, mask[:, d_i, d_i, :], time,  # input of long interest model
                                                user[:, s:e], poi[:, s:e], itv[:, s:e], dist[:, s:e])  # of short

                        acc_tra += calculate_true_positive(prob_train, new_tar)  # calculate the accurate
                        mrr_tra += mean_reciprocal_rank(prob_train, new_tar)
                        # re_prob, re_tar = regenerate_prob(prob_train, new_tar, self.neg)  # build the negative
                        # loss_train = self.criterion(re_prob, re_tar)
                        loss_train = self.criterion(prob_train, new_tar)

                        loss_show += loss_train.item()
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    # validation
                    elif i < len(data_index)-gap_valid_test:

                        if epoch == 0:
                            self.save_index_val.append(d_i)
                        self.model.eval()
                        valid_size += self.bs

                        with torch.no_grad():
                            prob_val = self.model(user, poi, mask[:, d_i, d_i, :], time,  # input of the long model
                                                  user[:, s:e], poi[:, s:e], itv[:, s:e], dist[:, s:e])  # of short
                            acc_val += calculate_true_positive(prob_val, new_tar)
                            mrr_val += mean_reciprocal_rank(prob_val, new_tar)
                    # test
                    else:

                        if epoch == 0:
                            self.save_index_tes.append(d_i)
                        self.model.eval()
                        test_size += self.bs

                        with torch.no_grad():
                            prob_test = self.model(user, poi, mask[:, d_i, d_i, :], time,  # input of long model
                                                   user[:, s:e], poi[:, s:e], itv[:, s:e], dist[:, s:e])  # of short
                            acc_tes += calculate_true_positive(prob_test, new_tar)
                            mrr_tes += mean_reciprocal_rank(prob_test, new_tar)
                    bar.update(1)
                woyaonulecaocaocao[i_batch] = acc_tes[0] - old_acc
                if woyaonulecaocaocao[i_batch] <= cut_limit and self.bs == 1:
                    wocaonima.append(user[:, 0].item())
            scheduler.step()

            p_los = loss_show / train_size / len(dataset)
            p_atr = (np.array(acc_tra) / train_size) * 100
            p_avl = (np.array(acc_val) / valid_size) * 100
            p_ate = (np.array(acc_tes) / test_size) * 100
            p_mtr = mrr_tra / train_size * 100
            p_mvl = mrr_val / valid_size * 100
            p_mte = mrr_tes / test_size * 100
            print(f'----loss:{p_los:.8f}')
            print(f'----train_ac:[{p_atr[0]:.2f}%, {p_atr[1]:.2f}%, {p_atr[2]:.2f}%, {p_atr[3]:.2f}%],'
                  f'  train_MRR: [{p_mtr:.2f}%]')
            print(f'----valid_ac:[{p_avl[0]:.2f}%, {p_avl[1]:.2f}%, {p_avl[2]:.2f}%, {p_avl[3]:.2f}%],'
                  f'  valid_MRR: [{p_mvl:.2f}%]')
            print(f'----test__ac:[{p_ate[0]:.2f}%, {p_ate[1]:.2f}%, {p_ate[2]:.2f}%, {p_ate[3]:.2f}%],'
                  f'  test_MRR: [{p_mte:.2f}%]')
            print(woyaonulecaocaocao)
            print(wocaonima)
            bar.close()

            # update and save model state

            # if self.threshold < np.mean(acc_tes):
            #     self.threshold = np.mean(acc_tes)
            if self.threshold < acc_tes[0]:
                self.threshold = acc_tes[0]

                # save the model
                self.save_acc_val.append(acc_val)
                self.save_acc_tes.append(acc_tes)
                self.save_epoch.append(e + 1)
                torch.save({'state_dict': self.model.state_dict(),
                            'model': self.model,
                            'index_val': self.save_index_val,
                            'index_tes': self.save_index_tes,
                            'acc_val': self.save_acc_val,
                            'acc_tes': self.save_acc_tes,
                            'epoch': self.save_epoch}, self.model_path)

    def re_run(self, user_seq, poi_seq, mask, time_seq, interval_seq, distance_seq, best_model_info):

        self.model.load_state_dict(best_model_info['state_dict'])
        # self.model = best_model_info['model']
        self.model.eval()
        index_val = best_model_info['index_val']
        index_tes = best_model_info['index_tes']
        valid_size = user_seq.shape[0]*self.eva_size
        test_size = user_seq.shape[0]*self.eva_size

        dataset = SsmDataSet(user_seq, poi_seq, mask, time_seq, interval_seq, distance_seq, self.device)
        loader = DataLoader(dataset, batch_size=self.bs, shuffle=False)

        tar_val = torch.zeros((self.bs, self.poi_num)).to(self.device)
        tar_tes = torch.zeros((self.bs, self.poi_num)).to(self.device)
        acc_val, acc_tes = [0, 0, 0, 0], [0, 0, 0, 0]

        bar = tqdm(total=valid_size/self.bs, desc='--calculating')
        for count, (user, poi, mask, time, itv, dist) in enumerate(loader):

            for i in range(self.eva_size):
                tar_val.fill_(0)
                tar_tes.fill_(0)
                v_i = index_val[i+count*self.eva_size]
                t_i = index_tes[i+count*self.eva_size]

                # validation
                s = max(0, v_i - self.short_len + 1)  # start index of the short interest model input
                e = v_i + 1  # end index of the short interest model input
                prob_val = self.model(user, poi, mask[:, v_i, v_i, :], time,  # input of the long interest model
                                      user[:, s:e], poi[:, s:e], itv[:, s:e], dist[:, s:e])  # of short

                for b in range(self.bs):  # build target
                    tar_val[b, poi[b, v_i + 1] - 1] = 1

                acc_val += calculate_true_positive(prob_val, tar_val)

                # test
                s = max(0, t_i - self.short_len + 1)  # start index of the short interest model input
                e = t_i + 1  # end index of the short interest model input
                prob_test = self.model(user, poi, mask[:, t_i, t_i, :], time,  # input of the long interest model
                                       user[:, s:e], poi[:, s:e], itv[:, s:e], dist[:, s:e])  # of short

                for b in range(self.bs):  # build target
                    tar_tes[b, poi[b, t_i + 1] - 1] = 1

                acc_tes += calculate_true_positive(prob_test, tar_tes)
                bar.update(1)

        p_avl = (np.array(acc_val) / valid_size) * 100
        p_ate = (np.array(acc_tes) / test_size) * 100
        print(f'----valid_ac:[{p_avl[0]:.2f}%, {p_avl[1]:.2f}%, {p_avl[2]:.2f}%, {p_avl[3]:.2f}%]')
        print(f'----test__ac:[{p_ate[0]:.2f}%, {p_ate[1]:.2f}%, {p_ate[2]:.2f}%, {p_ate[3]:.2f}%]')
        bar.close()


if __name__ == '__main__':

    # get the data provider
    sb = Seqs_Build(return_root_mat(arguments))

    sb.user = torch.tensor(sb.user, dtype=torch.long)
    sb.poi = torch.tensor(sb.poi, dtype=torch.long)
    sb.mask = torch.tensor(sb.mask).clone().detach().bool()
    sb.time = torch.tensor(sb.time, dtype=torch.long)
    sb.interval = torch.tensor(sb.interval, dtype=torch.long)
    sb.distance = torch.tensor(sb.distance, dtype=torch.long)

    moma = ModelManager(sb.poi_num, sb.long_seq_length, sb.user_emb_num, sb.poi_emb_num, sb.dis_emb_num, arguments)

    model_path = 'data/' + arguments.dataset + '/model_ssm.pth'
    if not os.path.exists(model_path):
        print('Step 2 : start training')
        moma.run(sb.user, sb.poi, sb.mask, sb.time, sb.interval, sb.distance)

    if os.path.exists(model_path):
        print(f'Step 2 : The model for {arguments.dataset} dataset has been trained')
        checkpoint = torch.load(model_path)
        # moma.run(sb.user, sb.poi, sb.mask, sb.time, sb.interval, sb.distance, checkpoint['model'])
        moma.re_run(sb.user, sb.poi, sb.mask, sb.time, sb.interval, sb.distance, checkpoint)
